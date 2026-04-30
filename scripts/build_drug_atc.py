"""
Build a Generic-Name → ATC (Levels 1-4) mapping using NIH/NLM RxNav APIs and
upload to Hugging Face.

Pipeline:
1. Pull unique Generic Name values from the HF prescriber dataset (cached).
2. Fetch the full ATC1-4 class catalog from RxClass once (code → name).
3. For each generic name try in order:
   a. Exact name → /rxcui.json?name=...
   b. Salt-stripped name → /rxcui.json?name=...
   c. Aggressively normalized name (CMS abbreviations expanded, separators
      flattened, parentheticals dropped, biosimilar suffix dropped)
      → /rxcui.json?name=...
   d. Same normalized name → /Rxcui/approximateTerm.json
4. With the resulting CUI, fetch ATC1-4 classes via /rxclass/class/byRxcui.json
   - if empty, drill to ingredient CUIs via /rxcui/{cui}/related.json?tty=IN
     and retry per ingredient
5. Pick the most specific class (longest classId) and derive Levels 1-3 by
   truncating that code; look up each level's name in the catalog.
6. Pull the canonical RxNorm name for the matched CUI for QC.
7. Write drug_atc.parquet (matched, lean) and drug_atc_qc.csv (all rows incl.
   unmatched, with method + RxNorm name + status). Upload parquet to HF.

Caching: every API response is written to hf_staging/rxnav_cache/.
Re-runs are nearly instant after the first warm-up.

Run:
    python scripts/build_drug_atc.py
"""

from __future__ import annotations

import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
STAGING = REPO_DIR / "hf_staging"
CACHE_DIR = STAGING / "rxnav_cache"
GENERICS_CACHE = STAGING / "unique_generics.json"
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"

DEFAULT_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", DEFAULT_DATASET_ID)
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/prescribers"

RXNAV = "https://rxnav.nlm.nih.gov/REST"
RATE_LIMIT_DELAY = 0.1  # ~10 req/sec on live calls

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY", "PR", "VI", "GU", "AS", "MP",
]

# Tokens stripped wherever they appear as a whole word during normalization.
# Includes salts, formulation descriptors, and biosimilar-type qualifiers.
SALT_TOKENS = {
    "hydrochloride", "hcl",
    "sulfate", "sulphate", "sulf",
    "phosphate", "phos",
    "sodium", "sod",
    "calcium", "ca",
    "potassium", "k",
    "magnesium", "mg",
    "tartrate", "succinate", "succ", "maleate", "fumarate", "fum",
    "citrate", "acetate", "acet", "lactate",
    "besylate", "besilate", "bes",
    "mesylate", "tosylate",
    "bromide", "brom", "chloride", "nitrate",
    "carbonate", "bicarbonate",
    "hydrobromide", "hbr",
    "monohydrate", "dihydrate", "trihydrate",
    "dipropionate", "propionate",
    "valerate", "benzoate", "benz",
    "lysine", "glycine", "arginine",  # amino-acid co-formulants used as salts
    "medoxomil", "lauroxil",  # ester / acyl prodrug suffixes
    "na",  # sodium abbreviation as standalone token
    "pf",  # preservative-free
    "submicr",  # truncation of submicronized
    "submicronized", "nanocrystallized", "microspheres", "liposomal",
    "nonrefrigerated", "lyophilized",
    "human", "recombinant",
    "extended", "release", "er", "xr", "sr", "ir", "la", "od",
    "preservative", "free",
    "with", "and", "in",
}

# Map common CMS abbreviations / truncations to their canonical full form.
# Applied per whitespace-separated token after lowercasing.
CMS_ALIASES = {
    "hcthiazid": "hydrochlorothiazide",
    "hydrochlorothiazid": "hydrochlorothiazide",  # CMS sometimes drops trailing 'e'
    "hctz": "hydrochlorothiazide",
    "chlorthal": "chlorthalidone",
    "caff": "caffeine",
    "dihydrocod": "dihydrocodeine",
    "lamivudi": "lamivudine",
    "med": "medoxomil",  # mapped here so SALT_TOKENS catches it on the next pass
    "elag": "elagolix",
    "clav": "clavulanate",
    "amox": "amoxicillin",
    "cefur": "cefuroxime",
    "tazo": "tazobactam",
    "sulb": "sulbactam",
    "sulbac": "sulbactam",
    "polym": "polymyxin",
    "trimethop": "trimethoprim",
    "tmp": "trimethoprim",
    "smx": "sulfamethoxazole",
    "smz": "sulfamethoxazole",
    "doluteg": "dolutegravir",
    # Frequently-truncated combination components
    "pseudoephed": "pseudoephedrine",
    "dm": "dextromethorphan",
    "glycopyr": "glycopyrronium",  # INN; RxNorm uses this in combo names
    "butalb": "butalbital",
    "bictegrav": "bictegravir",
    "emtricit": "emtricitabine",
    "tenofov": "tenofovir",
    "metronid": "metronidazole",
}

# Regexes used by normalize_cms.
RE_PAREN = re.compile(r"\([^)]*\)")
RE_BIOSIM_SUFFIX = re.compile(r"-[a-z]{4}\b")
RE_PCT = re.compile(r"\b\d+(?:\.\d+)?\s*%")  # "5%", "0.9 %"
RE_IN_DILUENT = re.compile(
    r"\bin\s+\d*(?:\.\d+)?\s*%?\s*(sod chlor|sodium chloride|dextrose|polysorbat\w*|d\d+w|nacl|saline|ethanol|water|glycine)\b"
)
RE_NONALPHA = re.compile(r"[^a-z0-9 -]")
RE_WS = re.compile(r"\s+")


def load_token() -> str:
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "rb") as f:
            secrets = tomllib.load(f)
        token = secrets.get("HF_TOKEN")
        if token:
            return token
    raise SystemExit("HF_TOKEN not set in env or .streamlit/secrets.toml")


def fetch_unique_generics() -> list[str]:
    if GENERICS_CACHE.exists():
        with open(GENERICS_CACHE) as f:
            cached = json.load(f)
        print(f"Loaded {len(cached):,} unique generic names from cache")
        return cached

    print("Pulling unique Generic Name values from HF prescriber dataset…")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    seen: set[str] = set()
    for state in US_STATES:
        for year in (2021, 2022, 2023):
            url = f"{HF_BASE}/year={year}/State={state}/data_0.parquet"
            try:
                rows = con.execute(
                    f'SELECT DISTINCT "Generic Name" FROM read_parquet(\'{url}\') '
                    f'WHERE "Generic Name" IS NOT NULL'
                ).fetchall()
                seen.update(r[0] for r in rows if r[0])
            except duckdb.IOException:
                continue
    names = sorted(n.strip() for n in seen if n and n.strip())
    print(f"  {len(names):,} unique generic names")
    GENERICS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(GENERICS_CACHE, "w") as f:
        json.dump(names, f)
    return names


def safe_filename(name: str) -> str:
    return urllib.parse.quote(name, safe="").replace("/", "_")[:200]


def cached_get(cache_path: Path, url: str) -> dict | None:
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            cache_path.unlink()

    time.sleep(RATE_LIMIT_DELAY)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def strip_salt(name: str) -> str:
    parts = name.lower().split()
    while parts and parts[-1] in SALT_TOKENS:
        parts.pop()
    return " ".join(parts)


def normalize_cms(name: str) -> str:
    """Aggressive normalization for combinations and CMS quirks."""
    s = name.lower()
    s = RE_PAREN.sub(" ", s)               # drop parentheticals
    s = RE_BIOSIM_SUFFIX.sub("", s)        # drop FDA biosimilar suffix like -epbx
    s = RE_PCT.sub(" ", s)                 # drop percentage tokens
    s = RE_IN_DILUENT.sub(" ", s)          # drop "in 0.9% sod chlor" and friends
    s = s.replace("/", " ").replace(",", " ")
    # split-only on hyphens that look like separators (between two letter tokens)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", s)
    s = RE_NONALPHA.sub(" ", s)
    parts = [p for p in RE_WS.sub(" ", s).split() if p]
    # expand aliases, then drop salt/qualifier tokens
    expanded = []
    for p in parts:
        p = CMS_ALIASES.get(p, p)
        if p not in SALT_TOKENS:
            expanded.append(p)
    return " ".join(expanded).strip()


def find_rxcui(name: str) -> str | None:
    if not name:
        return None
    url = f"{RXNAV}/rxcui.json?name={urllib.parse.quote(name)}"
    cache_path = CACHE_DIR / "rxcui" / f"{safe_filename(name)}.json"
    data = cached_get(cache_path, url)
    if not data:
        return None
    ids = (data.get("idGroup") or {}).get("rxnormId") or []
    return ids[0] if ids else None


def approximate_rxcui(name: str) -> str | None:
    if not name:
        return None
    url = f"{RXNAV}/Rxcui/approximateTerm.json?term={urllib.parse.quote(name)}&maxEntries=1"
    cache_path = CACHE_DIR / "approx" / f"{safe_filename(name)}.json"
    data = cached_get(cache_path, url)
    if not data:
        return None
    candidates = (data.get("approximateGroup") or {}).get("candidate") or []
    return candidates[0]["rxcui"] if candidates else None


def get_rxnorm_name(cui: str) -> str | None:
    url = f"{RXNAV}/rxcui/{cui}/property.json?propName=RxNorm%20Name"
    cache_path = CACHE_DIR / "rxnorm_name" / f"{cui}.json"
    data = cached_get(cache_path, url)
    if not data:
        return None
    items = (
        ((data.get("propConceptGroup") or {}).get("propConcept")) or []
    )
    if items:
        return items[0].get("propValue")
    return None


def get_atc_classes(rxcui: str) -> list[dict]:
    url = f"{RXNAV}/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC"
    cache_path = CACHE_DIR / "atc" / f"{rxcui}.json"
    data = cached_get(cache_path, url)
    if not data:
        return []
    items = (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or []
    out = []
    for it in items:
        concept = it.get("rxclassMinConceptItem") or {}
        if concept.get("classType") == "ATC1-4":
            out.append({"classId": concept["classId"], "className": concept["className"]})
    return out


def get_ingredient_cuis(rxcui: str) -> list[str]:
    url = f"{RXNAV}/rxcui/{rxcui}/related.json?tty=IN"
    cache_path = CACHE_DIR / "related_in" / f"{rxcui}.json"
    data = cached_get(cache_path, url)
    if not data:
        return []
    out = []
    for group in (data.get("relatedGroup") or {}).get("conceptGroup") or []:
        for prop in group.get("conceptProperties") or []:
            cui = prop.get("rxcui")
            if cui and cui != rxcui:
                out.append(cui)
    return out


def fetch_all_atc_classes() -> dict[str, str]:
    print("Fetching ATC1-4 class catalog from RxClass…")
    url = f"{RXNAV}/rxclass/allClasses.json?classTypes=ATC1-4"
    cache_path = CACHE_DIR / "all_atc1_4.json"
    data = cached_get(cache_path, url)
    out: dict[str, str] = {}
    if data:
        items = (data.get("rxclassMinConceptList") or {}).get("rxclassMinConcept") or []
        for it in items:
            if it.get("classType") == "ATC1-4":
                out[it["classId"]] = it["className"]
    print(f"  {len(out):,} ATC classes loaded")
    return out


def pick_primary_class(classes: list[dict], prefer_combo: bool) -> dict:
    """Pick the most relevant ATC class from RxNav's list.

    For combinations, prefer a class whose name signals it's a combination
    (e.g. "Combinations of oral blood glucose lowering drugs"). For single
    ingredients, prefer a class that does NOT signal combination so plain
    bupropion isn't tagged as an antiobesity drug just because Contrave
    exists. RxClass often duplicates the primary class in its response (one
    entry per relating concept), so we tie-break by occurrence count.
    """
    has_combo_word = lambda c: "combin" in c["className"].lower()
    if prefer_combo:
        subset = [c for c in classes if has_combo_word(c)]
    else:
        subset = [c for c in classes if not has_combo_word(c)]
    if not subset:
        subset = classes

    # Score each unique classId by frequency, then prefer longer classId
    counts: dict[str, int] = {}
    by_id: dict[str, dict] = {}
    for c in subset:
        cid = c["classId"]
        counts[cid] = counts.get(cid, 0) + 1
        by_id[cid] = c
    best_id = max(by_id, key=lambda cid: (counts[cid], len(cid)))
    return by_id[best_id]


def derive_levels(class_id: str, class_map: dict[str, str]) -> dict:
    lengths = {1: 1, 2: 3, 3: 4, 4: 5}
    out = {}
    for level, length in lengths.items():
        if len(class_id) >= length:
            code = class_id[:length]
            out[f"atc_level_{level}_code"] = code
            out[f"atc_level_{level}_name"] = class_map.get(code)
        else:
            out[f"atc_level_{level}_code"] = None
            out[f"atc_level_{level}_name"] = None
    return out


def resolve_to_atc(name: str, class_map: dict[str, str]) -> tuple[dict | None, str]:
    """Return (record_dict, status). status one of: matched | no_rxcui | no_atc."""
    cui = find_rxcui(name)
    method = "exact"
    if not cui:
        stripped = strip_salt(name)
        if stripped and stripped != name.lower():
            cui = find_rxcui(stripped)
            if cui:
                method = "salt_stripped"
    norm = normalize_cms(name) if not cui else None
    if not cui and norm and norm != name.lower():
        cui = find_rxcui(norm)
        if cui:
            method = "normalized"
    if not cui and norm:
        # RxNorm combination canonical form: alphabetically-sorted ingredients
        # joined by " / ", e.g. "metformin / sitagliptin".
        words = norm.split()
        if len(words) >= 2:
            combo = " / ".join(sorted(words))
            cui = find_rxcui(combo)
            if cui:
                method = "combo_alphabetical"
    if not cui and norm:
        cui = approximate_rxcui(norm)
        if cui:
            method = "approximate"
    if not cui:
        return None, "no_rxcui"

    classes = get_atc_classes(cui)
    if not classes:
        # Drill to ingredient CUIs and retry
        for ing_cui in get_ingredient_cuis(cui):
            ing_classes = get_atc_classes(ing_cui)
            if ing_classes:
                classes = ing_classes
                cui = ing_cui  # report the ingredient CUI (it's the one that mapped)
                method = method + "+ingredient_drill"
                break

    if not classes:
        return {"rxcui": cui, "match_method": method}, "no_atc"

    primary = pick_primary_class(classes, prefer_combo="combo" in method)
    levels = derive_levels(primary["classId"], class_map)
    rxnorm_name = get_rxnorm_name(cui)
    return (
        {"rxcui": cui, "rxnorm_name": rxnorm_name, "match_method": method, **levels},
        "matched",
    )


def main() -> None:
    token = load_token()
    STAGING.mkdir(parents=True, exist_ok=True)

    generics = fetch_unique_generics()
    class_map = fetch_all_atc_classes()
    if not class_map:
        raise SystemExit("Failed to fetch ATC class catalog from RxClass.")

    qc_rows: list[dict] = []
    matched_rows: list[dict] = []
    n = len(generics)
    print(f"\nResolving {n:,} generic names via RxNav…")
    t0 = time.time()
    counts = {"matched": 0, "no_rxcui": 0, "no_atc": 0}
    for i, name in enumerate(generics, 1):
        if i % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"  {i:,}/{n:,} matched={counts['matched']:,} "
                f"no_rxcui={counts['no_rxcui']:,} no_atc={counts['no_atc']:,} "
                f"({elapsed:.0f}s)"
            )
        record, status = resolve_to_atc(name, class_map)
        counts[status] += 1
        qc = {"generic_name": name, "status": status}
        if record:
            qc.update(record)
        qc_rows.append(qc)
        if status == "matched":
            matched_rows.append({"Generic Name": name, **record})

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.0f}s. matched={counts['matched']:,}/{n:,} "
        f"({100 * counts['matched'] / n:.1f}%), no_rxcui={counts['no_rxcui']:,}, "
        f"no_atc={counts['no_atc']:,}"
    )

    matched_df = pd.DataFrame(matched_rows)
    out_path = STAGING / "drug_atc.parquet"
    matched_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    qc_df = pd.DataFrame(qc_rows)
    qc_csv = STAGING / "drug_atc_qc.csv"
    qc_cols = [
        "generic_name", "status", "match_method", "rxcui", "rxnorm_name",
        "atc_level_4_code", "atc_level_4_name",
        "atc_level_3_code", "atc_level_3_name",
        "atc_level_2_code", "atc_level_2_name",
        "atc_level_1_code", "atc_level_1_name",
    ]
    for c in qc_cols:
        if c not in qc_df.columns:
            qc_df[c] = None
    qc_df = qc_df[qc_cols].sort_values(["status", "generic_name"], ascending=[False, True])
    qc_df.to_csv(qc_csv, index=False)
    print(f"Wrote {qc_csv} ({qc_csv.stat().st_size / 1024:.1f} KB, {len(qc_df):,} rows)")

    api = HfApi(token=token)
    print(f"\nUploading drug_atc.parquet → {HF_DATASET_ID}/drug_atc.parquet")
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="drug_atc.parquet",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Refresh drug → ATC mapping (v2: combos + ingredient drill)",
    )
    print(f"Done → https://huggingface.co/datasets/{HF_DATASET_ID}/blob/main/drug_atc.parquet")


if __name__ == "__main__":
    main()
