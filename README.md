# Medicare Part D Prescribing Dashboard

Interactive Streamlit dashboard for exploring Medicare Part D prescribing patterns across 2021, 2022, and 2023.

The app summarizes prescription drug cost and utilization by year, drug, specialty, state, and prescriber. It can build optimized parquet cache files from raw Medicare Part D CSV, TXT, TSV, or parquet files on first run.

## Features

- Global filters for year, state, specialty, brand name, and generic name
- Total drug cost and total claims summary cards
- Top drugs by year, grouped by brand or generic name
- Top specialties by year
- Yearly trend charts for selected drugs
- Prescriber search and yearly prescriber trend charts
- Local parquet cache generation for faster repeat loading

## Data

This project expects Medicare Part D source files for 2021, 2022, and 2023 in the repository root. The app detects files whose names include one of those years and whose extension is `.csv`, `.txt`, `.tsv`, or `.parquet`.

The raw CMS CSV files and generated parquet caches are intentionally ignored by git because they are very large. Keep them locally when running the dashboard, but do not commit them to GitHub.

Expected raw column names include:

- `Prscrbr_NPI`
- `Prscrbr_First_Name`
- `Prscrbr_Last_Org_Name`
- `Prscrbr_State_Abrvtn`
- `Prscrbr_Type`
- `Brnd_Name`
- `Gnrc_Name`
- `Tot_Clms`
- `Tot_30day_Fills`
- `Tot_Day_Suply`
- `Tot_Drug_Cst`
- `Tot_Benes`

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

Run the dashboard:

```bash
streamlit run Med_D_dashboard.py
```

## Project Structure

```text
.
├── Med_D_dashboard.py
├── medicare_D_dashboard.ipynb
├── requirements.txt
├── data/
│   └── processed/
└── README.md
```

## Cache Files

On first run, the app creates parquet cache files under `data/processed/`:

- `medicare_partd_2021_2023.parquet`
- `medicare_partd_prescriber_2021_2023.parquet`
- `medicare_partd_prescriber_year_trends_2021_2023.parquet`
- `medicare_partd_prescriber_index_2021_2023.parquet`

These files are generated artifacts and are not tracked by git.
