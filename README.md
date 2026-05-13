# Medicare Drug Dashboards

Interactive Streamlit app for exploring Medicare Part D prescribing patterns and Medicare Part B drug spending.

The app has four pages, accessible from a top navigation bar:

- **Med D Drugs Dashboard** — aggregate analysis of pharmacy-dispensed drug costs, claims, and specialty patterns
- **Med B Drugs Dashboard** — annual Medicare Part B (clinician-administered) drug spending by HCPCS code, brand, and generic
- **Med B Drugs by State & Provider Specialty** — state-aware Medicare Part B drug dashboard sourced from CMS Physician PUF rendering-provider data
- **Provider Search** — drill down to individual Part D prescribers or Part B rendering providers by city, radius, or name

The shared **Ask AI** assistant can query the aggregate Part D, Part B, and Physician PUF dashboard rollups. Individual provider-level radius/name searches remain in the Provider Search page controls.

The UI uses a shared healthcare analytics design system with top navigation, light page headers, soft chart cards, modern table styling, and a prominent Ask AI command area.

Data is hosted on Hugging Face and downloaded once per container into a local cache, so cold starts pay a small one-time cost and subsequent reads are local-disk speed.

## Features

### Med D Drugs Dashboard

- Global filters for year, state, and specialty, plus a Brand / Generic grouping toggle
- Total drug cost and total claims summary cards
- Part D cost totals are gross CMS `Tot_Drug_Cst` values from the provider-drug detail file; they are before rebates and are not Medicare net program spending
- Top drugs by year (brand or generic), with bar / treemap toggle
- Top specialties by year
- Yearly trend charts for selected drugs
- Modern chart-detail tables are collapsed behind focused "View detailed rows" expanders to reduce dashboard clutter
- US state choropleth with a **Total Cost / Per Capita Cost** toggle (Census Bureau Vintage 2023 population estimates)
- **Ask AI** chatbot (Claude Haiku 4.5) that writes DuckDB SQL across the shared dashboard rollups and explains results in plain English

### Med B Drugs Dashboard

- Segmented year selector and Brand / Generic grouping toggle (defaults to brand)
- Four metric cards: total spend, total claims, distinct drugs, and first→last-year spend growth
- Total yearly spending chart with year-over-year change callouts
- Top drugs by spend (Bar / Treemap toggle) with per-section Top N control (5 / 10 / 20)
- **Top specialties by spend** (Bar / Treemap toggle), built from CMS Physician PUF drug-HCPCS rows
- Yearly trend chart for selected drugs
- Modern HCPCS-level drill-down table (one row per HCPCS code × brand × year) with formatted metrics and spend-share context
- Indigo header banner and multi-hue chart palettes that keep the page visually distinct from Part D

### Med B Drugs by State & Provider Specialty

- 2021-2023 Physician PUF drug-HCPCS dashboard with segmented year selector, state filter, and Brand / Generic grouping
- Same metric-card and chart structure as the national Med B dashboard, using Total Services instead of Total Claims
- Total yearly spending, top drugs, top rendering specialties, yearly trends, and HCPCS-level drill-down filtered to selected states
- Modern detail tables use consistent formatting, summary chips, and spend-share context across dashboards
- Caveats called out in-app: totals exclude facility-billed administrations and beneficiary counts are summed across rendering providers

### Provider Search

- **Part D / Part B toggle** at the top of the page
- City + Radius (5/10/25/50/100 mi) + Drug → top prescribers within a centroid-distance radius
- Provider Name → top drugs for matching prescribers or rendering providers
- Hover help explains the centroid-based, all-or-nothing-per-city radius behavior
- Part B drug input accepts brand (e.g. Keytruda), generic (e.g. pembrolizumab), HCPCS code (e.g. J9271), or HCPCS description; ranked result tables show Brand Name and Generic Name columns. Brand/Generic mapping is joined at query time from `part_b_drug_spending.parquet`.

## Data Architecture

Source of truth for all production data is the public Hugging Face dataset [`mbateya/medicare_part_d_prescribers`](https://huggingface.co/datasets/mbateya/medicare_part_d_prescribers). The app fetches files via `huggingface_hub.hf_hub_download`, which caches under `~/.cache/huggingface/hub/`, so each file is downloaded at most once per container.

| File on HF | Size | Purpose |
|---|---|---|
| `processed/medicare_partd_2021_2023.parquet` | 69 MB | Aggregated rows used by the Med D dashboard and Ask AI's Part D DuckDB view |
| `processed/medicare_partd_top_providers_by_drug_2021_2023.parquet` | 14 MB | Top-providers-by-drug summary used by the Dashboard |
| `prescribers/year=YYYY/State=XX/data_0.parquet` | partitioned ~78M rows | Per-prescriber rows used by Provider Search Part D mode |
| `partb_prescribers/year=YYYY/State=XX/data_0.parquet` | partitioned ~1.5M rows | Per-rendering-provider Part B rows (drug HCPCS only); built from CMS Physician PUF; powers Provider Search Part B mode. Note: "Prescriber NPI/Name" columns are kept for partition-naming parallelism with Part D, but the values are the *rendering* clinician (who administered the drug and billed Medicare), not necessarily the ordering clinician. The UI surfaces these as "Rendering Provider" / "Rendering NPI". |
| `partb_drug_by_specialty.parquet` | ~50 KB | Year × Specialty × HCPCS rollup for the Med B Drugs Dashboard's Top Specialties chart |
| `partb_drug_by_state.parquet` | ~452 KB | Year × State × HCPCS rollup with Brand/Generic joined; powers Med B Drugs by State & Provider Specialty |
| `partb_drug_by_state_specialty.parquet` | ~1.4 MB | Year × State × Specialty × HCPCS rollup for the Med B Drugs by State & Provider Specialty Top Specialties chart |
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
├── Med_D_dashboard.py              # Med D Drugs Dashboard page
├── dashboard_ai.py                 # shared Ask AI dialog + DuckDB rollup views
├── dashboard_design.py             # shared visual system: headers, cards, chart shells
├── dashboard_tables.py             # shared modern table rendering + column formatting helpers
├── build_provider_summary.py       # offline ETL: top-providers-by-drug rollup
├── pages/
│   ├── 1_Provider_Search.py        # Provider Search page
│   ├── 2_Part_B_Drugs.py           # Med B Drugs Dashboard page
│   └── 3_Med_B_Drugs_State.py      # Med B Drugs by State & Provider Specialty page
├── data/
│   ├── drug_atc_overrides.csv      # tracked manual ATC overrides
│   └── processed/                  # gitignored local parquet cache (optional)
├── scripts/
│   ├── build_hf_prescriber_dataset.py   # builds the HF prescribers/year=…/State=… partitions
│   ├── upload_processed_to_hf.py        # uploads aggregated dashboard parquets to HF
│   ├── build_geocoded_cities.py         # builds cities.parquet from GeoNames
│   ├── build_state_population.py        # builds state_population.parquet from Census Vintage 2023
│   ├── build_drug_atc.py                # builds drug_atc.parquet via NLM RxNav + overrides
│   ├── build_part_b_drug_spending.py    # builds part_b_drug_spending.parquet from CMS Part B
│   └── build_partb_prescriber_dataset.py # builds partb_prescribers/ + Part B Physician PUF rollups
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
| `scripts/build_partb_prescriber_dataset.py` | Annually, when CMS releases new run-year Physician & Other Practitioners (Provider × Service) files. Downloads ~6 GB of CSVs, filters to drug HCPCS, partitions per state-year, and uploads provider partitions plus Year×Specialty×HCPCS, Year×State×HCPCS, and Year×State×Specialty×HCPCS rollups. Use `--rollup-only` to rebuild/upload rollups from existing local partitions. Update `CMS_PUF_URLS` in the script when CMS publishes new files. |

All scripts read `HF_TOKEN` from the environment or `.streamlit/secrets.toml`. The drug-ATC builder caches every RxNav response under `hf_staging/rxnav_cache/`, so iterative re-runs after editing the override CSV are nearly instant.
