"""
Build the Medicare Part B Drug Spending parquet from CMS public data.

Source: CMS "Medicare Part B Spending by Drug" annual file
        https://data.cms.gov (dataset 76a714ad-3a2c-43ac-b76d-9dadf8f7d890).
        Wide format: one row per drug (HCPCS + brand + generic), columns
        repeated per year. We melt to long format: one row per
        (HCPCS, Brand, Generic, Year).

Output: hf_staging/part_b_drug_spending.parquet
        Uploaded to mbateya/medicare_part_d_prescribers/part_b_drug_spending.parquet.

Run:
    python scripts/build_part_b_drug_spending.py
"""

from __future__ import annotations

import io
import os
import re
import time
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

# Most recent published file (RY2025, covers 2019-2023). Update annually
# when CMS releases a new run year. Find the latest URL by browsing
# https://data.cms.gov/data-api/v1/dataset/76a714ad-3a2c-43ac-b76d-9dadf8f7d890/data-viewer
CMS_URL = (
    "https://data.cms.gov/sites/default/files/2025-05/"
    "f52d5fcd-8d93-481d-9173-6219813e4efb/"
    "DSD_PTB_RY25_P06_V10_DYT23_HCPCS-%20250430.csv"
)

PER_YEAR_FIELDS = {
    "Tot_Spndng": "Total Spending",
    "Tot_Dsg_Unts": "Total Dosage Units",
    "Tot_Clms": "Total Claims",
    "Tot_Benes": "Total Beneficiaries",
    "Avg_Spndng_Per_Dsg_Unt": "Avg Spending per Dosage Unit",
    "Avg_Spndng_Per_Clm": "Avg Spending per Claim",
    "Avg_Spndng_Per_Bene": "Avg Spending per Beneficiary",
    "Outlier_Flag": "Outlier Flag",
}

ID_COLS = ["HCPCS_Cd", "HCPCS_Desc", "Brnd_Name", "Gnrc_Name"]
ID_RENAME = {
    "HCPCS_Cd": "HCPCS Code",
    "HCPCS_Desc": "HCPCS Description",
    "Brnd_Name": "Brand Name",
    "Gnrc_Name": "Generic Name",
}


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


def fetch_csv() -> pd.DataFrame:
    print(f"Downloading {CMS_URL} …")
    t0 = time.time()
    with urllib.request.urlopen(CMS_URL) as resp:
        raw = resp.read()
    print(f"  {len(raw) / 1024:.0f} KB in {time.time() - t0:.1f}s")
    return pd.read_csv(io.BytesIO(raw))


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide CMS file (per-year columns) to long format (one row per drug-year)."""
    # Discover available years from column names like Tot_Spndng_2019, Tot_Spndng_2023
    years = sorted({
        int(m.group(1))
        for col in df.columns
        for m in [re.match(r"Tot_Spndng_(\d{4})$", col)]
        if m
    })
    print(f"Years detected: {years}")

    long_frames = []
    for year in years:
        cols = {f"{prefix}_{year}": clean for prefix, clean in PER_YEAR_FIELDS.items()
                if f"{prefix}_{year}" in df.columns}
        sub = df[ID_COLS + list(cols.keys())].copy()
        sub = sub.rename(columns={**cols, **ID_RENAME})
        sub["Year"] = year
        long_frames.append(sub)
    long = pd.concat(long_frames, ignore_index=True)

    # Drop rows where the drug has no data for that year
    long = long[long["Total Spending"].notna()].copy()
    # Coerce numerics
    for c in ["Total Spending", "Total Dosage Units", "Total Claims", "Total Beneficiaries",
              "Avg Spending per Dosage Unit", "Avg Spending per Claim",
              "Avg Spending per Beneficiary"]:
        if c in long.columns:
            long[c] = pd.to_numeric(long[c], errors="coerce")
    if "Outlier Flag" in long.columns:
        long["Outlier Flag"] = pd.to_numeric(long["Outlier Flag"], errors="coerce").fillna(0).astype(int)

    # Reorder columns
    front = ["Year", "HCPCS Code", "HCPCS Description", "Brand Name", "Generic Name"]
    rest = [c for c in long.columns if c not in front]
    return long[front + rest].sort_values(["Year", "Total Spending"], ascending=[True, False])


def upload(token: str, path: Path) -> None:
    api = HfApi(token=token)
    print(f"Uploading {path.name} → {HF_DATASET_ID}/{path.name}")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Add/refresh Medicare Part B drug spending parquet",
    )
    print(f"Done → https://huggingface.co/datasets/{HF_DATASET_ID}/blob/main/{path.name}")


def main() -> None:
    token = load_token()
    STAGING.mkdir(parents=True, exist_ok=True)

    df = fetch_csv()
    print(f"Raw: {len(df):,} rows × {len(df.columns)} cols")
    long = melt_to_long(df)
    print(f"Long: {len(long):,} rows ({long['Year'].nunique()} years × ~{len(df)} drugs)")

    out = STAGING / "part_b_drug_spending.parquet"
    long.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    # Summary
    by_year = long.groupby("Year")["Total Spending"].sum().sort_index()
    print("\nTotal Part B drug spend by year:")
    for y, v in by_year.items():
        print(f"  {y}: ${v / 1e9:.2f}B")

    upload(token, out)


if __name__ == "__main__":
    main()
