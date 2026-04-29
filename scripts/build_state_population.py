"""
Build a state-population lookup parquet from US Census Bureau Vintage 2023
estimates and upload it to Hugging Face. Used by the dashboard to compute
per-capita drug spending on the choropleth map.

Source: NST-EST2023-ALLDATA.csv from the Census PEP program.
Columns kept: State (full name), POPESTIMATE2021/2022/2023.

Run:
    python scripts/build_state_population.py
"""

from __future__ import annotations

import io
import os
import tomllib
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
STAGING = REPO_DIR / "hf_staging"
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"

DEFAULT_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", DEFAULT_DATASET_ID)

CENSUS_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/state/totals/NST-EST2023-ALLDATA.csv"
)
STATE_SUMLEV = 40  # row-level summary: state


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


def fetch_population() -> pd.DataFrame:
    print(f"Downloading {CENSUS_URL} …")
    with urllib.request.urlopen(CENSUS_URL) as resp:
        raw = resp.read()
    print(f"  {len(raw) / 1024:.1f} KB")

    df = pd.read_csv(io.BytesIO(raw))
    states = df[df["SUMLEV"] == STATE_SUMLEV][
        ["NAME", "POPESTIMATE2021", "POPESTIMATE2022", "POPESTIMATE2023"]
    ].rename(
        columns={
            "NAME": "State",
            "POPESTIMATE2021": "Population_2021",
            "POPESTIMATE2022": "Population_2022",
            "POPESTIMATE2023": "Population_2023",
        }
    )
    print(f"  {len(states)} state rows (incl. DC)")
    return states.reset_index(drop=True)


def upload(token: str, path: Path) -> None:
    api = HfApi(token=token)
    print(f"Uploading {path.name} → {HF_DATASET_ID}/{path.name}")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Add state population lookup",
    )
    print(f"Done → https://huggingface.co/datasets/{HF_DATASET_ID}/blob/main/{path.name}")


def main() -> None:
    token = load_token()
    STAGING.mkdir(parents=True, exist_ok=True)

    pop = fetch_population()
    out = STAGING / "state_population.parquet"
    pop.to_parquet(out, index=False)
    print(f"\nWrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    print(pop.head(5).to_string(index=False))

    upload(token, out)


if __name__ == "__main__":
    main()
