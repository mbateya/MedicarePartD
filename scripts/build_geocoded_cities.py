"""
Build a (State, City) → (Latitude, Longitude) lookup parquet for the
Medicare Part D prescriber dataset and upload it to Hugging Face.

Process:
1. Pull unique (State, City) pairs from the HF prescriber dataset.
2. Download GeoNames US.zip (~70MB) for canonical place coordinates.
3. Match cities by normalized name within each state, picking the
   highest-population candidate when multiple places share a name.
4. Write hf_staging/cities.parquet, upload to the HF dataset root.

Run:
    python scripts/build_geocoded_cities.py
"""

from __future__ import annotations

import io
import os
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
STAGING = REPO_DIR / "hf_staging"
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"

DEFAULT_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", DEFAULT_DATASET_ID)
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/prescribers"

GEONAMES_URL = "https://download.geonames.org/export/dump/US.zip"

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY", "PR", "VI", "GU", "AS", "MP",
]


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


def fetch_unique_cities() -> pd.DataFrame:
    print("Pulling unique (State, City) from HF prescriber dataset…")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    frames: list[pd.DataFrame] = []
    for s in US_STATES:
        url = f"{HF_BASE}/year=2023/State={s}/data_0.parquet"
        try:
            df = con.execute(
                f"SELECT DISTINCT State, City FROM read_parquet('{url}') WHERE City <> 'Unknown'"
            ).fetchdf()
            frames.append(df)
        except duckdb.IOException:
            # State partition may not exist (territories with no rows in 2023)
            continue
    cities = pd.concat(frames, ignore_index=True).drop_duplicates(["State", "City"])
    print(f"  {len(cities):,} unique (State, City) pairs")
    return cities


def fetch_geonames() -> pd.DataFrame:
    print(f"Downloading GeoNames US.zip from {GEONAMES_URL} …")
    t0 = time.time()
    resp = urllib.request.urlopen(GEONAMES_URL)
    data = resp.read()
    print(f"  downloaded {len(data) / 1024 / 1024:.1f} MB in {time.time() - t0:.1f}s")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("US.txt") as f:
            gn = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=[
                    "geonameid", "name", "asciiname", "alternatenames",
                    "latitude", "longitude", "feature_class", "feature_code",
                    "country_code", "cc2", "admin1_code", "admin2_code",
                    "admin3_code", "admin4_code", "population", "elevation",
                    "dem", "timezone", "modification_date",
                ],
                dtype={"admin1_code": str},
                low_memory=False,
            )
    gn = gn[gn["feature_class"] == "P"].copy()
    gn = gn[["name", "asciiname", "latitude", "longitude", "admin1_code", "population"]]
    print(f"  GeoNames populated places: {len(gn):,}")
    return gn


def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .str.lower()
        .str.replace(r"\b(township|charter township|city|town|village|cdp|borough)\b", "", regex=True)
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def match(cities: pd.DataFrame, gn: pd.DataFrame) -> pd.DataFrame:
    cities["city_norm"] = normalize_name(cities["City"])
    gn["name_norm"] = normalize_name(gn["name"])
    gn["asciiname_norm"] = normalize_name(gn["asciiname"])

    # Exact normalized match on either name or asciiname; prefer larger population
    gn_long = pd.concat(
        [
            gn[["admin1_code", "name_norm", "latitude", "longitude", "population"]]
            .rename(columns={"name_norm": "key"}),
            gn[["admin1_code", "asciiname_norm", "latitude", "longitude", "population"]]
            .rename(columns={"asciiname_norm": "key"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    gn_long = gn_long.sort_values("population", ascending=False).drop_duplicates(
        ["admin1_code", "key"], keep="first"
    )

    out = cities.merge(
        gn_long,
        left_on=["State", "city_norm"],
        right_on=["admin1_code", "key"],
        how="left",
    )
    out = out[["State", "City", "latitude", "longitude"]].rename(
        columns={"latitude": "Latitude", "longitude": "Longitude"}
    )
    matched = out["Latitude"].notna().sum()
    print(f"  matched {matched:,} of {len(out):,} ({matched / len(out) * 100:.1f}%)")
    return out


def upload(token: str, path: Path) -> None:
    api = HfApi(token=token)
    print(f"Uploading {path.relative_to(REPO_DIR)} → {HF_DATASET_ID}/{path.name}")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Add geocoded cities lookup",
    )
    print(f"Done → https://huggingface.co/datasets/{HF_DATASET_ID}/blob/main/{path.name}")


def main() -> None:
    token = load_token()
    STAGING.mkdir(parents=True, exist_ok=True)

    cities = fetch_unique_cities()
    gn = fetch_geonames()
    geocoded = match(cities, gn)

    out_path = STAGING / "cities.parquet"
    geocoded.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Quick sanity check
    sample = geocoded[geocoded["City"].str.contains("Canton", case=False, na=False)]
    print("\nSample (cities containing 'Canton'):")
    print(sample.head(10).to_string(index=False))

    upload(token, out_path)


if __name__ == "__main__":
    main()
