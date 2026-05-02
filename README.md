# Medicare Part D Prescribing Dashboard

Interactive Streamlit app for exploring Medicare Part D prescribing patterns across 2021, 2022, and 2023.

The app has three pages, accessible from a top navigation bar:

- **Part D Dashboard** — aggregate analysis of pharmacy-dispensed drug costs, claims, and specialty patterns
- **Part B Drugs** — annual Medicare Part B (clinician-administered) drug spending by HCPCS code, brand, and generic
- **Provider Search** — drill down to individual Part D prescribers by city, radius, or name

Data is hosted on Hugging Face and downloaded once per container into a local cache, so cold starts pay a small one-time cost and subsequent reads are local-disk speed.

## Features

### Dashboard

- Global filters for year, state, specialty, brand name, and generic name
- Total drug cost and total claims summary cards
- Top drugs by year (brand or generic), with bar / treemap toggle
- Top specialties by year
- Yearly trend charts for selected drugs
- US state choropleth with a **Total Cost / Per Capita Cost** toggle (Census Bureau Vintage 2023 population estimates)
- **Ask AI** chatbot (Claude Haiku 4.5) that writes DuckDB SQL against the dataset and explains results in plain English

### Part B Drugs

- Year filter and optional generic-name filter
- KPI strip: total Part B drug spend, total claims, total beneficiaries, distinct drugs
- Top N drugs by spend (Bar / Treemap toggle, with distinct purple palette so it doesn't blur with Part D)
- Yearly trend chart for selected drugs
- HCPCS-level drill-down table (one row per HCPCS code × brand × year)

### Provider Search

- City + Drug → top prescribers in matching cities
- City + Radius (5/10/25/50/100 mi) + Drug → top prescribers within a centroid-distance radius
- Provider Name → top drugs prescribed by matching providers
- Hover help explains the centroid-based, all-or-nothing-per-city radius behavior

## Data Architecture

Source of truth for all production data is the public Hugging Face dataset [`mbateya/medicare_part_d_prescribers`](https://huggingface.co/datasets/mbateya/medicare_part_d_prescribers). The app fetches files via `huggingface_hub.hf_hub_download`, which caches under `~/.cache/huggingface/hub/`, so each file is downloaded at most once per container.

| File on HF | Size | Purpose |
|---|---|---|
| `processed/medicare_partd_2021_2023.parquet` | 69 MB | Aggregated rows used by the Dashboard and Ask AI's DuckDB view |
| `processed/medicare_partd_top_providers_by_drug_2021_2023.parquet` | 14 MB | Top-providers-by-drug summary used by the Dashboard |
| `prescribers/year=YYYY/State=XX/data_0.parquet` | partitioned ~78M rows | Per-prescriber rows used by Provider Search (city + drug, radius, provider name) |
| `cities.parquet` | 403 KB | (State, City) → (Latitude, Longitude) lookup for radius search; built from GeoNames |
| `state_population.parquet` | 4.5 KB | Census Bureau Vintage 2023 state populations for per-capita map |
| `drug_atc.parquet` | 82 KB | Generic Name → WHO ATC Levels 1-4 codes/names; built from NLM RxNav |
| `part_b_drug_spending.parquet` | 229 KB | Annual Medicare Part B drug spending by HCPCS code, brand, generic (long format, 2019-2023) |

Local-only artifacts (gitignored):

- `data/processed/*.parquet` — optional dev cache; if present, the app prefers these over HF downloads
- `MUP_DPR_*.csv` — raw CMS files used only by the offline build scripts
- `hf_staging/` — output of build scripts before upload
- `~/.cache/huggingface/hub/` — automatic HF download cache

Source-controlled data:

- `data/drug_atc_overrides.csv` — manual ATC assignments for drugs the algorithmic pipeline can't resolve (TPN solutions, vaccines without ATC links, complex combos, etc.)

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure secrets at `.streamlit/secrets.toml` (gitignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."   # required for the Ask AI chatbot
HF_TOKEN          = "hf_..."        # required only for build scripts that upload to HF
```

Run the app:

```bash
streamlit run app.py
```

The first page load downloads ~83 MB of dashboard parquets from HF into the local cache. Provider Search downloads each state-year partition lazily on demand (~5-50 MB per state).

## Project Structure

```text
.
├── app.py                          # Streamlit entry point; declares pages + nav
├── Med_D_dashboard.py              # Dashboard page
├── build_provider_summary.py       # offline ETL: top-providers-by-drug rollup
├── pages/
│   ├── 1_Provider_Search.py        # Provider Search page
│   └── 2_Part_B_Drugs.py           # Part B drug spending page
├── data/
│   ├── drug_atc_overrides.csv      # tracked manual ATC overrides
│   └── processed/                  # gitignored local parquet cache (optional)
├── scripts/
│   ├── build_hf_prescriber_dataset.py   # builds the HF prescribers/year=…/State=… partitions
│   ├── upload_processed_to_hf.py        # uploads aggregated dashboard parquets to HF
│   ├── build_geocoded_cities.py         # builds cities.parquet from GeoNames
│   ├── build_state_population.py        # builds state_population.parquet from Census Vintage 2023
│   ├── build_drug_atc.py                # builds drug_atc.parquet via NLM RxNav + overrides
│   └── build_part_b_drug_spending.py    # builds part_b_drug_spending.parquet from CMS Part B
├── requirements.txt
└── README.md
```

## Streamlit Cloud Deployment

The deployed app entry point is `app.py`. Streamlit Cloud's app settings should have:

- **Main file path:** `app.py`
- **Python version:** 3.12 (or any 3.11+ that satisfies `requirements.txt`)
- **Secrets:** paste the contents of your local `.streamlit/secrets.toml` into the Secrets tab

## Rebuilding the HF dataset

The Hugging Face artifacts only change when the underlying data changes (annual CMS release, new provider-level partitions, or refreshed lookups). The relevant scripts are:

| Script | When to run |
|---|---|
| `scripts/build_hf_prescriber_dataset.py` | New year of CMS Part D Prescriber data |
| `build_provider_summary.py` | Same trigger; produces the top-providers-by-drug rollup |
| `scripts/upload_processed_to_hf.py` | After regenerating the rolled-up dashboard parquets locally |
| `scripts/build_geocoded_cities.py` | Rare — only if GeoNames updates significantly |
| `scripts/build_state_population.py` | Annually, when Census releases new population estimates |
| `scripts/build_drug_atc.py` | After adding new entries to `data/drug_atc_overrides.csv`, or annually for new drug approvals |
| `scripts/build_part_b_drug_spending.py` | Annually, when CMS releases the new run-year Part B Drug Spending file (around April-May). Update `CMS_URL` in the script to the new run-year URL. |

All scripts read `HF_TOKEN` from the environment or `.streamlit/secrets.toml`. The drug-ATC builder caches every RxNav response under `hf_staging/rxnav_cache/`, so iterative re-runs after editing the override CSV are nearly instant.
