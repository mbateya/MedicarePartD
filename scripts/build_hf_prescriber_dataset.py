"""
Build a city-enriched, partitioned prescriber dataset from the raw CMS
Medicare Part D Prescribers CSVs and upload it to a Hugging Face dataset.

Inputs:  MUP_DPR_2021.csv, MUP_DPR_2022.csv, MUP_DPR_2023.csv (repo root)
Output:  hf_staging/prescribers/year=YYYY/state=XX/data_0.parquet
Upload:  https://huggingface.co/datasets/<HF_DATASET_ID>

Configuration:
- HF_DATASET_ID — set via env var to override the default (e.g. "mbateya/medicare_part_d_prescribers")
- HF_TOKEN     — read from env var, falling back to .streamlit/secrets.toml

Run from the repo root:
    python scripts/build_hf_prescriber_dataset.py
"""

from __future__ import annotations

import os
import shutil
import time
import tomllib
from pathlib import Path

import duckdb
from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_DIR / "hf_staging"
PRESCRIBER_DIR = STAGING_ROOT / "prescribers"
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"

DEFAULT_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", DEFAULT_DATASET_ID)

YEARS = (2021, 2022, 2023)


def load_token() -> str:
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "rb") as f:
            secrets = tomllib.load(f)
        token = secrets.get("HF_TOKEN")
        if token:
            return token
    raise SystemExit(
        "HF_TOKEN not found. Set it in the environment or in "
        f"{SECRETS_FILE.relative_to(REPO_DIR)}"
    )


def build_year(con: duckdb.DuckDBPyConnection, year: int) -> int:
    csv_path = REPO_DIR / f"MUP_DPR_{year}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing raw CSV: {csv_path}")

    out_dir = PRESCRIBER_DIR / f"year={year}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{year}] reading {csv_path.name} → {out_dir.relative_to(REPO_DIR)}")
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT
                {year}::INTEGER AS Year,
                Prscrbr_State_Abrvtn AS State,
                COALESCE(NULLIF(TRIM(Prscrbr_City), ''), 'Unknown') AS City,
                CAST(Prscrbr_NPI AS VARCHAR) AS "Prescriber NPI",
                TRIM(
                    COALESCE(Prscrbr_First_Name, '') || ' ' ||
                    COALESCE(Prscrbr_Last_Org_Name, '')
                ) AS "Prescriber Name",
                CASE
                    WHEN Prscrbr_Type = 'Interventional Cardiology' THEN 'Cardiology'
                    WHEN Prscrbr_Type = 'Medical Oncology' THEN 'Hematology-Oncology'
                    ELSE Prscrbr_Type
                END AS Specialty,
                Brnd_Name AS "Brand Name",
                Gnrc_Name AS "Generic Name",
                Tot_Clms AS "Total Claims",
                Tot_30day_Fills AS "Total 30-Day Fills",
                Tot_Day_Suply AS "Total Days Supply",
                Tot_Drug_Cst AS "Total Drug Cost",
                Tot_Benes AS "Total Beneficiaries"
            FROM read_csv('{csv_path.as_posix()}', AUTO_DETECT=TRUE)
            WHERE Prscrbr_State_Abrvtn IS NOT NULL
              AND Brnd_Name IS NOT NULL
        ) TO '{out_dir.as_posix()}' (
            FORMAT PARQUET,
            PARTITION_BY (State),
            COMPRESSION ZSTD
        );
        """
    )
    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_dir.as_posix()}/**/*.parquet')"
    ).fetchone()[0]
    elapsed = time.time() - t0
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*.parquet")) / 1024 / 1024
    print(f"[{year}] {rows:,} rows · {size_mb:.1f} MB · {elapsed:.0f}s")
    return rows


def upload(token: str) -> None:
    api = HfApi(token=token)
    print(f"Uploading {STAGING_ROOT.relative_to(REPO_DIR)} → {HF_DATASET_ID}")
    api.upload_folder(
        folder_path=str(STAGING_ROOT),
        path_in_repo="",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Add city-enriched prescriber dataset (2021-2023)",
    )
    print(f"Done → https://huggingface.co/datasets/{HF_DATASET_ID}")


def main() -> None:
    token = load_token()
    PRESCRIBER_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    total = 0
    for year in YEARS:
        total += build_year(con, year)

    print(f"\nTotal rows: {total:,}")
    upload(token)


if __name__ == "__main__":
    main()
