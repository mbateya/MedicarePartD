"""
Pre-aggregate top providers per drug per year from the full prescriber parquet.
Reads row-group by row-group to avoid loading 77M rows at once.
State and Specialty are kept so global dashboard filters can be applied at query time.
Output: data/processed/medicare_partd_top_providers_by_drug_2021_2023.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_DIR = Path(__file__).resolve().parent
SRC = REPO_DIR / "data" / "processed" / "medicare_partd_prescriber_2021_2023.parquet"
DST = REPO_DIR / "data" / "processed" / "medicare_partd_top_providers_by_drug_2021_2023.parquet"

GROUP_COLS = ["Year", "Generic Name", "Brand Name", "Prescriber Name", "State", "Specialty"]
METRIC_COLS = [
    "Total Claims",
    "Total 30-Day Fills",
    "Total Days Supply",
    "Total Drug Cost",
    "Total Beneficiaries",
]
# Keep top 50 per (Year, Brand Name) to preserve enough rows after state/specialty filtering
TOP_N_PER_DRUG_YEAR = 50

pf = pq.ParquetFile(SRC)
n_groups = pf.metadata.num_row_groups
print(f"Processing {n_groups} row groups …")

partial: list[pd.DataFrame] = []

for i in range(n_groups):
    chunk = pf.read_row_group(i, columns=GROUP_COLS + METRIC_COLS).to_pandas()
    agg = chunk.groupby(GROUP_COLS, dropna=False, as_index=False)[METRIC_COLS].sum()
    partial.append(agg)
    if (i + 1) % 10 == 0 or i == n_groups - 1:
        print(f"  row groups processed: {i + 1}/{n_groups}")
        combined = pd.concat(partial, ignore_index=True)
        partial = [combined.groupby(GROUP_COLS, dropna=False, as_index=False)[METRIC_COLS].sum()]

summary = partial[0]
print(f"Full summary rows: {len(summary):,}")

# Identify top-50 providers per (Year, Brand Name) by total cost across all states/specialties
provider_totals = (
    summary.groupby(["Year", "Brand Name", "Prescriber Name"], as_index=False)["Total Drug Cost"]
    .sum()
)
top_providers = (
    provider_totals.sort_values("Total Drug Cost", ascending=False)
    .groupby(["Year", "Brand Name"], group_keys=False)
    .head(TOP_N_PER_DRUG_YEAR)[["Year", "Brand Name", "Prescriber Name"]]
)
top_key = top_providers.set_index(["Year", "Brand Name", "Prescriber Name"]).index

keep_mask = summary.set_index(["Year", "Brand Name", "Prescriber Name"]).index.isin(top_key)
top = summary[keep_mask].reset_index(drop=True)

top["Cost per Claim"] = top["Total Drug Cost"] / top["Total Claims"].replace(0, pd.NA)
top["Cost per 30-Day Fill"] = top["Total Drug Cost"] / top["Total 30-Day Fills"].replace(0, pd.NA)

print(f"Top-provider rows saved: {len(top):,}")
top.to_parquet(DST, index=False)
print(f"Saved → {DST}")
