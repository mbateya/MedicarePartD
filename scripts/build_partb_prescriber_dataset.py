"""
Build the Medicare Part B prescriber dataset from the CMS Physician PUF and
upload to Hugging Face.

Source: CMS "Medicare Physician & Other Practitioners — by Provider and Service"
        annual files (~2 GB each), filtered to HCPCS_Drug_Ind = 'Y' (drug rows
        only). Catalog: https://data.cms.gov/data.json (search title).

Outputs uploaded to mbateya/medicare_part_d_prescribers:
- partb_prescribers/year=YYYY/State=XX/data_0.parquet  — per-NPI per-HCPCS rows
- partb_drug_by_specialty.parquet                       — Year × Specialty × HCPCS rollup

Run from the repo root (downloads ~6 GB of raw CSVs to hf_staging/raw/, ~one-off):
    python scripts/build_partb_prescriber_dataset.py
"""

from __future__ import annotations

import os
import shutil
import time
import tomllib
import urllib.request
from pathlib import Path

import duckdb
from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_DIR / "hf_staging"
RAW_DIR = STAGING_ROOT / "raw"
PRESCRIBER_DIR = STAGING_ROOT / "partb_prescribers"
SPECIALTY_ROLLUP = STAGING_ROOT / "partb_drug_by_specialty.parquet"
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"

DEFAULT_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", DEFAULT_DATASET_ID)

# Resolved 2026-05 from CMS DCAT catalog (https://data.cms.gov/data.json).
# Update this dict when CMS releases new run-year files.
CMS_PUF_URLS = {
    2021: "https://data.cms.gov/sites/default/files/2025-11/bffaf97a-c2ab-4fd7-8718-be90742e3485/MUP_PHY_R25_P05_V20_D21_Prov_Svc.csv",
    2022: "https://data.cms.gov/sites/default/files/2025-11/53fb2bae-4913-48dc-a6d4-d8c025906567/MUP_PHY_R25_P05_V20_D22_Prov_Svc.csv",
    2023: "https://data.cms.gov/sites/default/files/2025-04/e3f823f8-db5b-4cc7-ba04-e7ae92b99757/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv",
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
    raise SystemExit(
        "HF_TOKEN not found. Set it in the environment or in "
        f"{SECRETS_FILE.relative_to(REPO_DIR)}"
    )


def download_csv(year: int, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"MUP_PHY_D{year % 100}_Prov_Svc.csv"
    if out.exists() and out.stat().st_size > 100_000_000:
        print(f"[{year}] cached {out.name} ({out.stat().st_size / 1e9:.2f} GB)")
        return out
    print(f"[{year}] downloading {url}")
    t0 = time.time()
    tmp = out.with_suffix(".csv.partial")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)
    tmp.rename(out)
    print(f"[{year}] {out.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    return out


def build_year(con: duckdb.DuckDBPyConnection, year: int, csv_path: Path) -> int:
    out_dir = PRESCRIBER_DIR / f"year={year}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{year}] filtering → {out_dir.relative_to(REPO_DIR)}")
    t0 = time.time()
    # CMS Physician PUF schema: Rndrng_NPI, Rndrng_Prvdr_*, HCPCS_*, Tot_*, Avg_Mdcr_Pymt_Amt
    # Filter HCPCS_Drug_Ind='Y' upfront. Same SPECIALTY_ALIASES as Med_D_dashboard.py.
    con.execute(
        f"""
        COPY (
            SELECT
                {year}::INTEGER AS Year,
                Rndrng_Prvdr_State_Abrvtn AS State,
                COALESCE(NULLIF(TRIM(Rndrng_Prvdr_City), ''), 'Unknown') AS City,
                CAST(Rndrng_NPI AS VARCHAR) AS "Prescriber NPI",
                TRIM(
                    COALESCE(Rndrng_Prvdr_First_Name, '') || ' ' ||
                    COALESCE(Rndrng_Prvdr_Last_Org_Name, '')
                ) AS "Prescriber Name",
                CASE
                    WHEN Rndrng_Prvdr_Type = 'Family Medicine' THEN 'Family Practice'
                    WHEN Rndrng_Prvdr_Type = 'Interventional Cardiology' THEN 'Cardiology'
                    WHEN Rndrng_Prvdr_Type = 'Medical Oncology' THEN 'Hematology-Oncology'
                    ELSE COALESCE(NULLIF(TRIM(Rndrng_Prvdr_Type), ''), 'Unknown')
                END AS Specialty,
                HCPCS_Cd AS "HCPCS Code",
                HCPCS_Desc AS "HCPCS Description",
                Tot_Benes AS "Total Beneficiaries",
                Tot_Srvcs AS "Total Services",
                Avg_Mdcr_Pymt_Amt AS "Avg Medicare Payment",
                (Tot_Srvcs * Avg_Mdcr_Pymt_Amt)::DOUBLE AS "Total Spending"
            FROM read_csv('{csv_path.as_posix()}', AUTO_DETECT=TRUE)
            WHERE HCPCS_Drug_Ind = 'Y'
              AND Rndrng_Prvdr_State_Abrvtn IS NOT NULL
              AND Rndrng_Prvdr_Cntry = 'US'
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
    print(f"[{year}] {rows:,} drug rows · {size_mb:.1f} MB · {elapsed:.0f}s")
    return rows


def build_specialty_rollup(con: duckdb.DuckDBPyConnection) -> int:
    """Year × Specialty × HCPCS Code rollup for the dashboard chart."""
    if SPECIALTY_ROLLUP.exists():
        SPECIALTY_ROLLUP.unlink()
    print(f"Building rollup → {SPECIALTY_ROLLUP.relative_to(REPO_DIR)}")
    con.execute(
        f"""
        COPY (
            SELECT
                Year,
                Specialty,
                "HCPCS Code",
                ANY_VALUE("HCPCS Description") AS "HCPCS Description",
                SUM("Total Spending") AS "Total Spending",
                SUM("Total Services") AS "Total Services",
                SUM("Total Beneficiaries") AS "Total Beneficiaries"
            FROM read_parquet('{PRESCRIBER_DIR.as_posix()}/**/*.parquet',
                              hive_partitioning = 1)
            GROUP BY Year, Specialty, "HCPCS Code"
        ) TO '{SPECIALTY_ROLLUP.as_posix()}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )
    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{SPECIALTY_ROLLUP.as_posix()}')"
    ).fetchone()[0]
    size_kb = SPECIALTY_ROLLUP.stat().st_size / 1024
    print(f"  rollup: {rows:,} rows · {size_kb:.0f} KB")
    return rows


def upload(token: str) -> None:
    api = HfApi(token=token)
    print(f"Uploading partb_prescribers/ → {HF_DATASET_ID}/partb_prescribers/")
    api.upload_folder(
        folder_path=str(PRESCRIBER_DIR),
        path_in_repo="partb_prescribers",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Add Med B prescriber dataset (2021-2023)",
    )
    print(f"Uploading {SPECIALTY_ROLLUP.name} → {HF_DATASET_ID}/{SPECIALTY_ROLLUP.name}")
    api.upload_file(
        path_or_fileobj=str(SPECIALTY_ROLLUP),
        path_in_repo=SPECIALTY_ROLLUP.name,
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="Add Med B drug-by-specialty rollup",
    )
    print(f"Done → https://huggingface.co/datasets/{HF_DATASET_ID}")


def main() -> None:
    token = load_token()
    PRESCRIBER_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4;")

    total = 0
    for year, url in CMS_PUF_URLS.items():
        csv_path = download_csv(year, url)
        total += build_year(con, year, csv_path)

    print(f"\nTotal drug-row records across years: {total:,}")
    build_specialty_rollup(con)

    # Spend summary for sanity-checking against the Med B Drugs Dashboard
    by_year = con.execute(
        f"""
        SELECT Year, SUM("Total Spending") AS spend
        FROM read_parquet('{PRESCRIBER_DIR.as_posix()}/**/*.parquet',
                          hive_partitioning = 1)
        GROUP BY Year ORDER BY Year
        """
    ).fetchall()
    print("\nTotal Part B drug spend by year (Physician PUF):")
    for y, v in by_year:
        print(f"  {y}: ${v / 1e9:.2f}B")

    upload(token)


if __name__ == "__main__":
    main()
