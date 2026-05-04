from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import anthropic
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots
import streamlit as st


APP_TITLE = "Medicare Part D Prescribing Dashboard"
APP_SUBTITLE = (
    "Interactive analysis of Medicare Part D prescribing patterns by year, drug, "
    "specialty, and state."
)

REPO_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = REPO_DIR / "data" / "processed" / "medicare_partd_2021_2023.parquet"
PROVIDER_SUMMARY_PATH = REPO_DIR / "data" / "processed" / "medicare_partd_top_providers_by_drug_2021_2023.parquet"

HF_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_PROCESSED_FILE = "processed/medicare_partd_2021_2023.parquet"
HF_PROVIDER_SUMMARY_FILE = "processed/medicare_partd_top_providers_by_drug_2021_2023.parquet"
HF_STATE_POPULATION_FILE = "state_population.parquet"
HF_DRUG_ATC_FILE = "drug_atc.parquet"


@st.cache_resource(show_spinner="Downloading processed dataset…")
def _resolved_processed_path() -> str:
    if PROCESSED_PATH.exists():
        return PROCESSED_PATH.as_posix()
    return hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_PROCESSED_FILE,
        repo_type="dataset",
    )


@st.cache_resource(show_spinner="Downloading provider data…")
def _resolved_provider_summary_path() -> str:
    if PROVIDER_SUMMARY_PATH.exists():
        return PROVIDER_SUMMARY_PATH.as_posix()
    return hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_PROVIDER_SUMMARY_FILE,
        repo_type="dataset",
    )


@st.cache_data(show_spinner="Loading state populations…", ttl=86400)
def load_state_population() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_STATE_POPULATION_FILE,
        repo_type="dataset",
    )
    return pd.read_parquet(path)


OTHERS_ATC_LABEL = "Others (unmapped)"


@st.cache_data(show_spinner="Loading drug → ATC mapping…", ttl=86400)
def load_drug_atc() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_DRUG_ATC_FILE,
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    return df[["Generic Name", "atc_level_2_code", "atc_level_2_name"]].copy()


def _attach_atc_level_2(df: pd.DataFrame) -> pd.DataFrame:
    atc = load_drug_atc()
    merged = df.merge(atc, on="Generic Name", how="left")
    merged["Therapeutic Class"] = (
        merged["atc_level_2_name"].fillna("").str.title().replace("", OTHERS_ATC_LABEL)
    )
    merged.loc[merged["atc_level_2_code"].isna(), "Therapeutic Class"] = OTHERS_ATC_LABEL
    return merged
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".parquet"}
TARGET_YEARS = {2021, 2022, 2023}

RAW_TO_CLEAN = {
    "Prscrbr_State_Abrvtn": "State",
    "Prscrbr_Type": "Specialty",
    "Brnd_Name": "Brand Name",
    "Gnrc_Name": "Generic Name",
    "Tot_Clms": "Total Claims",
    "Tot_30day_Fills": "Total 30-Day Fills",
    "Tot_Day_Suply": "Total Days Supply",
    "Tot_Drug_Cst": "Total Drug Cost",
}

RAW_REQUIRED_COLUMNS = list(RAW_TO_CLEAN.keys())
DIMENSION_COLUMNS = ["Year", "State", "Specialty", "Brand Name", "Generic Name"]
METRIC_COLUMNS = [
    "Total Claims",
    "Total 30-Day Fills",
    "Total Days Supply",
    "Total Drug Cost",
]
SPECIALTY_ALIASES = {
    "Family Medicine": "Family Practice",
    "Interventional Cardiology": "Cardiology",
    "Medical Oncology": "Hematology-Oncology",
}
# Brand-name canonicalization. CMS lists the same drug under multiple brand
# strings that differ only by injector / pen / pack size / inhaler tech /
# citrate-free formulation. Collapsing them gives true per-drug spend totals.
# Excluded on purpose: XR vs IR (different release profile), oral vs LAI
# (e.g. Abilify vs Abilify Maintena), and dosing-interval LAI variants
# (Invega Sustenna/Trinza/Hafyera).
BRAND_ALIASES = {
    # --- Biologics: pen / sureclick / syringe / pack-size variants ---
    "Humira Pen": "Humira",
    "Humira Pen Crohn's-Uc-Hs": "Humira",
    "Humira Pen Psor-Uveits-Adol Hs": "Humira",
    "Humira(Cf)": "Humira",
    "Humira(Cf) Pen": "Humira",
    "Humira(Cf) Pediatric Crohn's": "Humira",
    "Humira(Cf) Pen Crohn's-Uc-Hs": "Humira",
    "Humira(Cf) Pen Psor-Uv-Adol Hs": "Humira",
    "Enbrel Mini": "Enbrel",
    "Enbrel Sureclick": "Enbrel",
    "Dupixent Pen": "Dupixent",
    "Dupixent Syringe": "Dupixent",
    "Repatha Sureclick": "Repatha",
    "Repatha Pushtronex": "Repatha",
    "Repatha Syringe": "Repatha",
    "Cosentyx Pen": "Cosentyx",
    "Cosentyx Pen (2 Pens)": "Cosentyx",
    "Cosentyx (2 Syringes)": "Cosentyx",
    "Cosentyx Sensoready Pen": "Cosentyx",
    "Cosentyx Sensoready (2 Pens)": "Cosentyx",
    "Cosentyx Syringe": "Cosentyx",
    "Cosentyx Unoready Pen": "Cosentyx",
    # --- Insulins: pen / Solostar / Flextouch / Kwikpen / Penfill variants ---
    "Lantus Solostar": "Lantus",
    "Toujeo Max Solostar": "Toujeo",
    "Toujeo Solostar": "Toujeo",
    "Novolog Flexpen": "Novolog",
    "Novolog Penfill": "Novolog",
    "Humalog Junior Kwikpen": "Humalog",
    "Humalog Kwikpen U-100": "Humalog",
    "Humalog Kwikpen U-200": "Humalog",
    "Humalog Tempo Pen U-100": "Humalog",
    "Tresiba Flextouch U-100": "Tresiba",
    "Tresiba Flextouch U-200": "Tresiba",
    "Levemir Flexpen": "Levemir",
    "Levemir Flextouch": "Levemir",
    # --- Other delivery / pack variants ---
    "Restasis Multidose": "Restasis",
    "Victoza 2-Pak": "Victoza",
    "Victoza 3-Pak": "Victoza",
    "Tivicay Pd": "Tivicay",
    "Ingrezza Initiation Pack": "Ingrezza",
    # --- Inhaler-tech / HFA variants (same drug, different inhaler) ---
    "Spiriva Handihaler": "Spiriva",
    "Spiriva Respimat": "Spiriva",
    "Advair Hfa": "Advair Diskus",
    "Albuterol Sulfate Hfa": "Albuterol Sulfate",
    # --- CMS naming quirk ---
    "Levothyroxine": "Levothyroxine Sodium",
}
STATE_NAMES = {
    "AA": "Armed Forces Americas",
    "AE": "Armed Forces Europe",
    "AK": "Alaska",
    "AL": "Alabama",
    "AP": "Armed Forces Pacific",
    "AR": "Arkansas",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "FM": "Federated States of Micronesia",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MP": "Northern Mariana Islands",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VI": "U.S. Virgin Islands",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
    "XX": "Other",
    "ZZ": "Unknown",
}
STATE_NAME_TO_CODE = {v: k for k, v in STATE_NAMES.items()}
USA_MAP_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
}


st.markdown(
    """
<style>
/* Remove default Streamlit top padding */
.block-container { padding-top: 1.5rem !important; }

/* Style radio buttons as pill toggles */
div[role="radiogroup"] {
    display: flex;
    flex-direction: row;
    gap: 6px;
    flex-wrap: wrap;
}
div[role="radiogroup"] label {
    display: inline-flex;
    align-items: center;
    padding: 5px 14px;
    border-radius: 20px;
    border: 0.5px solid #d0d0d0;
    background: white;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
div[role="radiogroup"] label:has(input:checked) {
    border-color: #185fa5;
    color: #185fa5;
}
div[role="radiogroup"] input[type="radio"] { display: none; }

/* Style segmented_control selected button */
button[data-testid="stBaseButton-segmented_control"][aria-pressed="true"] {
    border-color: #185fa5 !important;
    color: #185fa5 !important;
}

/* Style multiselect chips as blue pills (not red) */
span[data-baseweb="tag"] {
    background-color: #1a3460 !important;
    border-color: #2a4a7a !important;
}
span[data-baseweb="tag"] span { color: #7fb3f5 !important; }

/* Hide the default metric delta arrow (we'll add our own) */
[data-testid="stMetricDelta"] svg { display: none; }

/* Remove Streamlit's default section dividers */
hr { border-color: #f0f0f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

DATAFRAME_CSS = """
<style>
[data-testid="stDataFrame"] table {
    font-size: 12px !important;
    border-collapse: collapse;
}
[data-testid="stDataFrame"] th {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #888 !important;
    background: #fafafa !important;
    border-bottom: 1px solid #e8e8e8 !important;
    padding: 8px 10px !important;
}
[data-testid="stDataFrame"] td {
    padding: 7px 10px !important;
    border-bottom: 0.5px solid #f3f3f3 !important;
    color: #222 !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: #f7f9ff !important;
}
div.filter-summary {
    margin-top: 0.85rem;
    margin-bottom: 0.5rem;
    padding: 0.95rem 1rem;
    background: #f8fbfc;
    border: 1px solid #e5eef2;
    border-radius: 16px;
}
div.filter-summary-title {
    color: #4e5a64;
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}
div.filter-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
span.filter-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.38rem 0.65rem;
    border-radius: 999px;
    background: #edf6f3;
    border: 1px solid #d6e9e1;
    color: #30584c;
    font-size: 0.84rem;
    font-weight: 600;
}
</style>
"""


def infer_year_from_filename(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        return None

    year = int(match.group(1))
    return year if year in TARGET_YEARS else None


def find_data_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_DIR.iterdir():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        year = infer_year_from_filename(path)
        if year in TARGET_YEARS and path.resolve() != PROCESSED_PATH.resolve():
            files.append(path)

    return sorted(files, key=lambda p: infer_year_from_filename(p) or 0)


def _delimiter_for(path: Path) -> str | None:
    if path.suffix.lower() == ".tsv":
        return "\t"
    if path.suffix.lower() == ".txt":
        return None
    return ","


def _read_raw_chunks(path: Path, chunksize: int = 300_000) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".parquet":
        try:
            yield pd.read_parquet(path, columns=RAW_REQUIRED_COLUMNS)
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet files requires pyarrow or fastparquet. Install pyarrow "
                "or provide CSV/TXT/TSV raw files."
            ) from exc
        return

    delimiter = _delimiter_for(path)
    read_kwargs = {
        "usecols": RAW_REQUIRED_COLUMNS,
        "chunksize": chunksize,
        "low_memory": False,
    }
    if delimiter is None:
        read_kwargs["sep"] = None
        read_kwargs["engine"] = "python"
    else:
        read_kwargs["sep"] = delimiter

    yield from pd.read_csv(path, **read_kwargs)


def _validate_columns(path: Path) -> None:
    if path.suffix.lower() == ".parquet":
        try:
            columns = pd.read_parquet(path).columns.tolist()
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet files requires pyarrow or fastparquet. Install pyarrow "
                "or provide CSV/TXT/TSV raw files."
            ) from exc
    else:
        delimiter = _delimiter_for(path)
        kwargs = {"nrows": 0}
        if delimiter is None:
            kwargs.update({"sep": None, "engine": "python"})
        else:
            kwargs["sep"] = delimiter
        columns = pd.read_csv(path, **kwargs).columns.tolist()

    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")


def _normalize_specialties(series: pd.Series) -> pd.Series:
    specialties = series.fillna("Unknown").astype(str).str.strip()
    specialties = specialties.mask(specialties.eq(""), "Unknown")
    return specialties.replace(SPECIALTY_ALIASES)


def _normalize_states(series: pd.Series) -> pd.Series:
    states = series.fillna("Unknown").astype(str).str.strip()
    states = states.mask(states.eq(""), "Unknown")
    return states.replace(STATE_NAMES)


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["State"] = _normalize_states(df["State"])
    df["Specialty"] = _normalize_specialties(df["Specialty"])
    df["Brand Name"] = df["Brand Name"].replace(BRAND_ALIASES)
    normalized = (
        df.groupby(DIMENSION_COLUMNS, dropna=False, as_index=False)[METRIC_COLUMNS]
        .sum()
    )
    return _add_derived_metrics(normalized)


def load_raw_files(files: list[Path]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError(
            "No Medicare Part D files for 2021, 2022, or 2023 were found in the repo folder."
        )

    year_frames = []
    for path in files:
        year = infer_year_from_filename(path)
        if year is None:
            continue

        _validate_columns(path)
        chunk_frames = []
        for chunk in _read_raw_chunks(path):
            chunk = chunk.rename(columns=RAW_TO_CLEAN)
            chunk["Year"] = year

            for col in ["State", "Specialty", "Brand Name", "Generic Name"]:
                chunk[col] = chunk[col].fillna("Unknown").astype(str).str.strip()
                chunk.loc[chunk[col].eq(""), col] = "Unknown"
            chunk["State"] = _normalize_states(chunk["State"])
            chunk["Specialty"] = _normalize_specialties(chunk["Specialty"])

            for col in METRIC_COLUMNS:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

            chunk_summary = (
                chunk.groupby(DIMENSION_COLUMNS, dropna=False, as_index=False)[METRIC_COLUMNS]
                .sum()
            )
            chunk_frames.append(chunk_summary)

        year_df = (
            pd.concat(chunk_frames, ignore_index=True)
            .groupby(DIMENSION_COLUMNS, dropna=False, as_index=False)[METRIC_COLUMNS]
            .sum()
        )
        year_frames.append(year_df)

    return pd.concat(year_frames, ignore_index=True)


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Cost per Claim"] = df["Total Drug Cost"] / df["Total Claims"].replace(0, pd.NA)
    df["Cost per 30-Day Fill"] = df["Total Drug Cost"] / df[
        "Total 30-Day Fills"
    ].replace(0, pd.NA)
    df[["Cost per Claim", "Cost per 30-Day Fill"]] = df[
        ["Cost per Claim", "Cost per 30-Day Fill"]
    ].fillna(0)
    return df


def _write_processed_parquet(df: pd.DataFrame, path: Path, error_action: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            f"{error_action} requires pyarrow or fastparquet. Install pyarrow before the first app run."
        ) from exc


def _load_processed_parquet(path: Path, action: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            f"{action} requires pyarrow or fastparquet. Install pyarrow or rebuild from CSV files."
        ) from exc


def build_processed_data() -> pd.DataFrame:
    files = find_data_files()
    df = load_raw_files(files)
    df = _normalize_dataset(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_processed_parquet(
        df,
        PROCESSED_PATH,
        "Saving the processed dataset",
    )
    return df


@st.cache_data(show_spinner="Loading Medicare Part D dataset...")
def load_or_build_dataset() -> pd.DataFrame:
    try:
        path = Path(_resolved_processed_path())
    except Exception:
        path = None

    if path is not None and path.exists():
        return _normalize_dataset(
            _load_processed_parquet(
                path,
                "Loading the processed parquet",
            )
        )

    return build_processed_data()


def render_top_n_control(label: str, key: str) -> int:
    selection = st.segmented_control(
        label,
        options=[5, 10, 20],
        default=10,
        key=key,
    )
    return int(selection or 10)


def apply_filters(
    df: pd.DataFrame,
    years: list[int],
    states: list[str],
    specialties: list[str],
) -> pd.DataFrame:
    filtered = df

    if years:
        filtered = filtered[filtered["Year"].isin(years)]
    if states:
        filtered = filtered[filtered["State"].isin(states)]
    if specialties:
        filtered = filtered[filtered["Specialty"].isin(specialties)]

    return filtered


def _summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    metric_cols = metric_cols or METRIC_COLUMNS
    summary = df.groupby(group_cols, dropna=False, as_index=False)[metric_cols].sum()
    return _add_derived_metrics(summary)


def _annual_top_n_full_history(
    df: pd.DataFrame,
    group_col: str,
    rank_col: str,
    top_n: int,
) -> pd.DataFrame:
    annual_top_values = (
        df.sort_values(["Year", rank_col], ascending=[True, False])
        .groupby("Year", group_keys=False)
        .head(top_n)[group_col]
        .drop_duplicates()
        .tolist()
    )
    top_values = (
        df[df[group_col].isin(annual_top_values)]
        .groupby(group_col, dropna=False)[rank_col]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    order = {value: index for index, value in enumerate(top_values)}
    top_df = df[df[group_col].isin(top_values)].copy()
    top_df["_Sort Order"] = top_df[group_col].map(order)
    return (
        top_df.sort_values(["_Sort Order", "Year"])
        .drop(columns="_Sort Order")
        .reset_index(drop=True)
    )


def summarize_top_drugs(df: pd.DataFrame, grouping: str, top_n: int) -> pd.DataFrame:
    drug_col = "Brand Name" if grouping == "Brand name" else "Generic Name"
    summary = _summarize(df, ["Year", drug_col])
    summary = summary.rename(columns={drug_col: "Drug Name"})
    return _annual_top_n_full_history(summary, "Drug Name", "Total Drug Cost", top_n)


def summarize_top_specialties(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    summary = _summarize(df, ["Year", "Specialty"])
    return _annual_top_n_full_history(summary, "Specialty", "Total Drug Cost", top_n)


def summarize_top_therapeutic_classes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top ATC Level 2 therapeutic groups; unmapped generics fall under 'Others'."""
    df = _attach_atc_level_2(df)
    summary = _summarize(df, ["Year", "Therapeutic Class"])
    return _annual_top_n_full_history(summary, "Therapeutic Class", "Total Drug Cost", top_n)


def summarize_yearly_spending(df: pd.DataFrame) -> pd.DataFrame:
    return _summarize(df, ["Year"]).sort_values("Year")


@st.cache_data(show_spinner="Loading provider data...")
def load_provider_summary() -> pd.DataFrame:
    df = pd.read_parquet(_resolved_provider_summary_path())
    df["Brand Name"] = df["Brand Name"].replace(BRAND_ALIASES)
    return df


def summarize_top_providers(
    df: pd.DataFrame,
    drug_name: str,
    top_n: int,
    years: list[int] | None = None,
    states: list[str] | None = None,
    specialties: list[str] | None = None,
) -> pd.DataFrame:
    drug_df = df[df["Brand Name"] == drug_name].copy()
    if years:
        drug_df = drug_df[drug_df["Year"].isin(years)]
    if states:
        drug_df = drug_df[drug_df["State"].isin(states)]
    if specialties:
        drug_df = drug_df[drug_df["Specialty"].isin(specialties)]
    drug_df["Prescriber Name"] = drug_df["Prescriber Name"].str.slice(0, 35)
    summary = drug_df.groupby(["Year", "Prescriber Name"], as_index=False)[
        ["Total Claims", "Total 30-Day Fills", "Total Days Supply",
         "Total Drug Cost", "Total Beneficiaries"]
    ].sum()
    summary = _add_derived_metrics(summary)
    return _annual_top_n_full_history(summary, "Prescriber Name", "Total Drug Cost", top_n)


def summarize_trends(df: pd.DataFrame, grouping: str, selected_drugs: list[str]) -> pd.DataFrame:
    drug_col = "Brand Name" if grouping == "Brand name" else "Generic Name"
    trend_df = df[df[drug_col].isin(selected_drugs)]
    summary = _summarize(trend_df, ["Year", drug_col])
    return summary.rename(columns={drug_col: "Drug Name"}).sort_values(["Drug Name", "Year"])


def format_tables(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    currency_cols = [
        col
        for col in ["Cost per Claim", "Cost per 30-Day Fill"]
        if col in df.columns
    ]
    number_cols = [
        col
        for col in ["Total Claims", "Total 30-Day Fills", "Total Days Supply"]
        if col in df.columns
    ]
    formats = {col: "${:,.2f}" for col in currency_cols}
    formats.update({col: "{:,.0f}" for col in number_cols})
    if "Total Drug Cost" in df.columns:
        formats["Total Drug Cost"] = fmt_currency
    return df.style.format(formats)


def _filter_context(
    years: list[int],
    states: list[str],
    specialties: list[str],
) -> str:
    parts = []
    if years:
        parts.append(f"Years: {', '.join(map(str, years))}")
    if states:
        parts.append(f"States: {', '.join(states[:4])}{'...' if len(states) > 4 else ''}")
    if specialties:
        parts.append(
            f"Specialties: {', '.join(specialties[:3])}{'...' if len(specialties) > 3 else ''}"
        )
    return " | ".join(parts) if parts else "All available records"


def section_heading(text: str) -> None:
    st.markdown(
        f"""
<div style="margin: 28px 0 4px;">
  <span style="font-size:18px;font-weight:600;color:#0a1628;">{text}</span>
  <div style="height:2px;width:32px;background:#378add;border-radius:1px;margin-top:5px;"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def insight_strip(text: str) -> None:
    st.markdown(
        f"""
<div style="
    background: #f0f6ff;
    border-left: 3px solid #378add;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #0c447c;
    margin: 8px 0 14px;
    line-height: 1.6;
">{text}</div>
""",
        unsafe_allow_html=True,
    )


def chart_card(fig) -> None:
    with st.container():
        st.markdown(
            '<div style="background:white;border:0.5px solid #e8e8e8;'
            'border-radius:12px;padding:18px 20px;margin-bottom:12px;">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def style_fig(fig, title: str = "", subtitle: str = ""):
    """Apply consistent modern styling to all Plotly figures."""
    title_text = (
        f"<b>{title}</b><br>"
        f"<span style='font-size:12px;color:#888;font-weight:400'>{subtitle}</span>"
        if title
        else ""
    )
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=15, color="#0a1628"),
            x=0,
            xanchor="left",
            pad=dict(l=0, b=8),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#444"),
        margin=dict(t=60 if title else 20, r=20, b=70, l=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="#e8e8e8",
            tickfont=dict(size=11, color="#888"),
        ),
        yaxis=dict(
            gridcolor="#f5f5f5",
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(size=11, color="#888"),
        ),
        colorway=["#185fa5", "#7f77dd", "#1d9e75", "#ef9f27", "#d85a30"],
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#e8e8e8",
            font=dict(size=12, color="#111"),
        ),
    )
    return fig


def _fmt_num(value: float, unit: float) -> str:
    return f"{value / unit:.1f}".rstrip("0").rstrip(".")


def _fmt_cost(value: float) -> str:
    if value >= 1e9:
        return f"${_fmt_num(value, 1e9)}B"
    return f"${_fmt_num(value, 1e6)}M"


def _fmt_count(value: float) -> str:
    if value >= 1e9:
        return f"{_fmt_num(value, 1e9)}B"
    return f"{_fmt_num(value, 1e6)}M"


def render_metric_cards(filtered_df: pd.DataFrame, drug_col: str) -> None:
    total_cost = filtered_df["Total Drug Cost"].sum()
    total_claims = filtered_df["Total Claims"].sum()
    total_fills = filtered_df["Total 30-Day Fills"].sum()
    avg_cost_per_claim = total_cost / total_claims if total_claims else 0
    unique_drugs = filtered_df[drug_col].nunique()
    years_sorted = sorted(filtered_df["Year"].dropna().astype(int).unique())

    if len(years_sorted) >= 2:
        cost_first = filtered_df[filtered_df["Year"] == years_sorted[0]]["Total Drug Cost"].sum()
        cost_last = filtered_df[filtered_df["Year"] == years_sorted[-1]]["Total Drug Cost"].sum()
        growth_pct = (cost_last - cost_first) / cost_first * 100 if cost_first else 0
        growth_str = f"{growth_pct:+.1f}%"
        growth_sub = f"{_fmt_cost(cost_first)} &rarr; {_fmt_cost(cost_last)}"
    else:
        growth_str = "N/A"
        growth_sub = "Select 2+ years"

    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:18px 0 6px;">

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#378add;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:6px;">Total drug cost</div>
    <div style="font-size:24px;font-weight:600;color:#111;line-height:1;">{_fmt_cost(total_cost)}</div>
    <div style="font-size:12px;color:#1d9e75;margin-top:4px;">All selected years</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#7f77dd;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:6px;">Total claims</div>
    <div style="font-size:24px;font-weight:600;color:#111;line-height:1;">{_fmt_count(total_claims)}</div>
    <div style="font-size:12px;color:#888;margin-top:4px;">All selected years</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#1d9e75;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:6px;">Avg cost per claim</div>
    <div style="font-size:24px;font-weight:600;color:#111;line-height:1;">${avg_cost_per_claim:,.2f}</div>
    <div style="font-size:12px;color:#888;margin-top:4px;">Across all drugs</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#ef9f27;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:6px;">Total 30-day fills</div>
    <div style="font-size:24px;font-weight:600;color:#111;line-height:1;">{_fmt_count(total_fills)}</div>
    <div style="font-size:12px;color:#888;margin-top:4px;">All selected years</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#d85a30;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:6px;">Unique drugs</div>
    <div style="font-size:24px;font-weight:600;color:#111;line-height:1;">{unique_drugs:,}</div>
    <div style="font-size:12px;color:#888;margin-top:4px;">Current drug grouping</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#185fa5;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:6px;">Cost growth (first&rarr;last yr)</div>
    <div style="font-size:24px;font-weight:600;color:#111;line-height:1;">{growth_str}</div>
    <div style="font-size:12px;color:#888;margin-top:4px;">{growth_sub}</div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )


def _section_title(base: str, grouping: str | None = None, context: str | None = None) -> str:
    title = base
    if grouping:
        title = f"{title} ({grouping.lower()})"
    if context and context != "All available records":
        title = f"{title}<br><sup>{context}</sup>"
    return title


def _build_billions_ticks(max_value: float) -> tuple[list[float], list[str]]:
    if max_value <= 0:
        return [0], ["$0.0B"]

    tick_max_b = max_value / 1e9
    step_b = max(0.5, round(tick_max_b / 5, 1))
    if step_b >= 1:
        step_b = float(int(step_b)) if step_b >= 2 else 1.0

    tick_vals_b = [0.0]
    current = step_b
    while current < tick_max_b * 1.05:
        tick_vals_b.append(round(current, 2))
        current += step_b

    if tick_vals_b[-1] < tick_max_b:
        tick_vals_b.append(round(max(tick_max_b, tick_vals_b[-1] + step_b), 2))

    tick_vals = [value * 1e9 for value in tick_vals_b]
    tick_text = [f"${value:,.1f}B" for value in tick_vals_b]
    return tick_vals, tick_text


def fmt_currency(val: float) -> str:
    for divisor, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if val >= divisor:
            s = f"{val / divisor:.1f}".rstrip("0").rstrip(".")
            return f"${s}{suffix}"
    return f"${val:.0f}"


DRUG_TREEMAP_PALETTE = ["#185fa5", "#378add", "#b5d4f4", "#7f77dd", "#1d9e75", "#ef9f27", "#d85a30"]
SPECIALTY_TREEMAP_PALETTE = ["#c0392b", "#e67e22", "#f39c12", "#d35400", "#cc6633", "#f5c178", "#a44a3f"]
THERAPEUTIC_TREEMAP_PALETTE = ["#118a7e", "#1d9e75", "#3aae8e", "#67c89c", "#16a085", "#27ae60", "#a8e2c1"]
DRUG_YEAR_PALETTE = ["#b5d4f4", "#378add", "#185fa5"]
SPECIALTY_YEAR_PALETTE = ["#f5c178", "#e67e22", "#c0392b"]
THERAPEUTIC_YEAR_PALETTE = ["#a8e2c1", "#3aae8e", "#118a7e"]


def render_treemap(
    df: pd.DataFrame,
    name_col: str,
    top_n: int,
    title: str,
    palette: list[str] | None = None,
):
    totals = (
        df.groupby(name_col, dropna=False)["Total Drug Cost"]
        .sum()
        .reset_index()
        .sort_values("Total Drug Cost", ascending=False)
    )
    top = totals.head(top_n).copy()
    others_cost = totals.iloc[top_n:]["Total Drug Cost"].sum()
    palette = palette or DRUG_TREEMAP_PALETTE
    colors = [palette[i % len(palette)] for i in range(len(top))]
    top_abbrev = [fmt_currency(v) for v in top["Total Drug Cost"]]

    if others_cost > 0:
        others_count = len(totals) - top_n
        others_label = f"Others ({others_count} more)"
        named_total = top["Total Drug Cost"].sum()
        grand_total = named_total + others_cost
        named_frac = round(named_total / grand_total, 4)
        others_frac = round(others_cost / grand_total, 4)

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[named_frac, others_frac],
            specs=[[{"type": "treemap"}, {"type": "treemap"}]],
            horizontal_spacing=0.005,
        )
        fig.add_trace(
            go.Treemap(
                labels=top[name_col].tolist(),
                parents=[""] * len(top),
                values=top["Total Drug Cost"].tolist(),
                customdata=top_abbrev,
                marker=dict(colors=colors),
                texttemplate="<b>%{label}</b><br>%{customdata}",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>Total Drug Cost: %{value:$,.0f}<extra></extra>",
                sort=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Treemap(
                labels=[others_label],
                parents=[""],
                values=[others_cost],
                customdata=[fmt_currency(others_cost)],
                marker=dict(colors=["#cccccc"]),
                texttemplate="<b>%{label}</b><br>%{customdata}",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>Total Drug Cost: %{value:$,.0f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    else:
        fig = go.Figure(
            go.Treemap(
                labels=top[name_col].tolist(),
                parents=[""] * len(top),
                values=top["Total Drug Cost"].tolist(),
                customdata=top_abbrev,
                marker=dict(colors=colors),
                texttemplate="<b>%{label}</b><br>%{customdata}",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>Total Drug Cost: %{value:$,.0f}<extra></extra>",
                sort=False,
            )
        )

    fig = style_fig(fig, title=title)
    fig.update_layout(showlegend=False, margin=dict(t=60, r=10, b=10, l=10))
    return fig


def render_state_choropleth(
    df: pd.DataFrame,
    title: str,
    metric: str = "Total Cost",
    population_year: int = 2023,
):
    state_totals = (
        df.groupby("State", dropna=False)["Total Drug Cost"]
        .sum()
        .reset_index()
    )
    state_totals["state_code"] = state_totals["State"].map(STATE_NAME_TO_CODE)
    map_df = state_totals[state_totals["state_code"].isin(USA_MAP_STATE_CODES)].copy()

    if map_df.empty:
        fig = go.Figure()
        fig = style_fig(fig, title=title)
        fig.add_annotation(
            text="No US state data for current filters",
            showarrow=False,
            font=dict(size=13, color="#888"),
        )
        return fig

    pop = load_state_population()
    pop_col = f"Population_{population_year}"
    if pop_col not in pop.columns:
        pop_col = "Population_2023"
    map_df = map_df.merge(
        pop[["State", pop_col]].rename(columns={pop_col: "Population"}),
        on="State",
        how="left",
    )
    map_df["Cost per Capita"] = map_df["Total Drug Cost"] / map_df["Population"]

    map_df["cost_label"] = map_df["Total Drug Cost"].apply(fmt_currency)
    map_df["per_capita_label"] = map_df["Cost per Capita"].apply(
        lambda v: f"${v:,.0f}" if pd.notna(v) else "n/a"
    )
    map_df["population_label"] = map_df["Population"].apply(
        lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a"
    )

    if metric == "Per Capita Cost":
        color_col = "Cost per Capita"
        cb_formatter = lambda v: f"${v:,.0f}"
    else:
        color_col = "Total Drug Cost"
        cb_formatter = fmt_currency

    fig = px.choropleth(
        map_df,
        locations="state_code",
        locationmode="USA-states",
        color=color_col,
        scope="usa",
        color_continuous_scale=[
            [0.0, "#f0f6ff"],
            [0.5, "#5b8dd9"],
            [1.0, "#0a1628"],
        ],
        hover_name="State",
        custom_data=["cost_label", "per_capita_label", "population_label"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b>"
            "<br>Total Drug Cost: %{customdata[0]}"
            "<br>Cost per Capita: %{customdata[1]}"
            f"<br>Population ({population_year}): "
            "%{customdata[2]}<extra></extra>"
        ),
        marker_line_color="white",
        marker_line_width=0.5,
    )

    max_val = float(map_df[color_col].max())
    cb_ticks = [max_val * i / 4 for i in range(5)]
    cb_labels = [cb_formatter(v) for v in cb_ticks]

    fig = style_fig(fig, title=title)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="", font=dict(size=11)),
            tickvals=cb_ticks,
            ticktext=cb_labels,
            tickfont=dict(size=10, color="#888"),
            thickness=10,
            len=0.7,
            outlinewidth=0,
        ),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
            landcolor="#fafafa",
            subunitcolor="white",
            showlakes=False,
            showcoastlines=False,
        ),
        margin=dict(t=60, r=10, b=10, l=10),
    )
    return fig


def render_charts(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    orientation: str = "v",
    year_palette: list[str] | None = None,
):
    chart_df = df.copy()
    if color == "Year":
        chart_df[color] = chart_df[color].astype(str)
    year_values = sorted(chart_df[color].dropna().unique().tolist()) if color == "Year" else []
    palette = year_palette or DRUG_YEAR_PALETTE
    color_map = {
        year: palette[index % len(palette)]
        for index, year in enumerate(year_values)
    }

    fig = px.bar(
        chart_df,
        x=x,
        y=y,
        color=color,
        barmode="group",
        orientation=orientation,
        template="plotly_white",
        color_discrete_map=color_map if color == "Year" else None,
        hover_data={
            "Total Drug Cost": ":$,.2f",
            "Total Claims": ":,.0f",
            "Total 30-Day Fills": ":,.0f",
            "Cost per Claim": ":$,.2f",
            "Cost per 30-Day Fill": ":$,.2f",
        },
    )
    fig = style_fig(fig, title=title)
    fig.update_layout(legend_title_text=color)
    category_axis = x if orientation == "v" else y
    category_order = chart_df[category_axis].drop_duplicates().tolist()
    if orientation == "v":
        fig.update_xaxes(categoryorder="array", categoryarray=category_order)
    else:
        fig.update_yaxes(categoryorder="array", categoryarray=category_order)

    axis_max = float(chart_df[y].max() if orientation == "v" else chart_df[x].max())
    tick_vals, tick_text = _build_billions_ticks(axis_max)
    if orientation == "v":
        fig.update_yaxes(tickvals=tick_vals, ticktext=tick_text, rangemode="tozero")
    else:
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, rangemode="tozero")
    return fig


def render_yearly_spending_chart(df: pd.DataFrame, title: str):
    fig = px.area(
        df,
        x="Year",
        y="Total Drug Cost",
        template="plotly_white",
        hover_data={
            "Total Drug Cost": ":$,.2f",
            "Total Claims": ":,.0f",
            "Total 30-Day Fills": ":,.0f",
            "Cost per Claim": ":$,.2f",
            "Cost per 30-Day Fill": ":$,.2f",
        },
    )
    fig.update_traces(
        mode="lines+markers",
        line=dict(color="#185fa5", width=3),
        marker=dict(size=9, color="#185fa5"),
        fillcolor="rgba(181, 212, 244, 0.42)",
    )
    fig = style_fig(fig, title=title)
    fig.update_layout(showlegend=False)
    yearly_df = df.reset_index(drop=True)
    y_min = float(yearly_df["Total Drug Cost"].min())
    y_max = float(yearly_df["Total Drug Cost"].max())
    if y_min == y_max:
        y_padding = y_max * 0.05 if y_max else 1
        y_range = [max(0, y_min - y_padding), y_max + y_padding]
    else:
        y_range = [y_min * 0.95, y_max * 1.05]

    for index, row in yearly_df.iterrows():
        if index == 0:
            continue
        prev = yearly_df.iloc[index - 1]["Total Drug Cost"]
        curr = row["Total Drug Cost"]
        if prev:
            pct = (curr - prev) / prev * 100
            fig.add_annotation(
                x=row["Year"],
                y=curr,
                text=f"{pct:+.1f}%",
                showarrow=False,
                yshift=14,
                font=dict(size=12, color="#185fa5"),
            )

    tick_vals, tick_text = _build_billions_ticks(float(df["Total Drug Cost"].max()))
    fig.update_xaxes(dtick=1, tickmode="linear")
    fig.update_yaxes(tickvals=tick_vals, ticktext=tick_text, range=y_range)
    return fig


def _select_options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


CHATBOT_MODEL = "claude-haiku-4-5"
CHATBOT_MAX_TOOL_CALLS = 5
CHATBOT_ROW_LIMIT = 50
CHATBOT_MAX_TOKENS = 4096

CHATBOT_SYSTEM_PROMPT = """\
You are a data analyst for the Medicare Part D Prescribing dashboard. \
Answer the user's questions by querying a DuckDB table called `medicare` via the run_sql tool.

Schema of `medicare` (years 2021-2023):
  Year (INTEGER) — 2021, 2022, or 2023
  State (VARCHAR) — full state name e.g. 'California'. Some rows are 'Unknown' or US territories.
  Specialty (VARCHAR) — prescriber specialty e.g. 'Cardiology', 'Internal Medicine'.
    Note: 'Interventional Cardiology' is grouped into 'Cardiology';
    'Medical Oncology' is grouped into 'Hematology-Oncology'.
  "Brand Name" (VARCHAR) — drug brand name (case-sensitive)
  "Generic Name" (VARCHAR) — drug generic name (case-sensitive)
  "Total Drug Cost" (DOUBLE) — total cost in USD
  "Total Claims" (BIGINT)
  "Total 30-Day Fills" (DOUBLE)
  "Cost per Claim" (DOUBLE)
  "Cost per 30-Day Fill" (DOUBLE)

Guidelines:
- Always wrap multi-word column names in double quotes.
- LIMIT your final SELECT to 50 rows.
- After receiving query results, write a 1-3 sentence answer in plain English.
- Do not invent column names or values that aren't in the schema above.

Drug name matching (CRITICAL):
- The user's term may be a brand name, a generic name, a partial name, a misspelling, \
or a name that has dosage/form suffixes in the data (e.g. "Dupixent" is stored as \
"Dupixent Pen" and "Dupixent Syringe"; "Rosuvastatin" is stored as "Rosuvastatin Calcium"; \
"Xeljanz" is stored as both "Xeljanz" and "Xeljanz XR").
- ALWAYS match drug names with ILIKE '%term%' on BOTH "Brand Name" AND "Generic Name". \
NEVER use `=` for drug names.
- ALWAYS aggregate (SUM / SUM / SUM) across all matching rows so brand/form variants combine \
into one total.
- Example: "How much was spent on Dupixent in 2023?" →
    SELECT SUM("Total Drug Cost") AS total_cost
    FROM medicare
    WHERE Year = 2023
      AND ("Brand Name" ILIKE '%Dupixent%' OR "Generic Name" ILIKE '%Dupixent%');
- Example: "spending on rosuvastatin in 2023" →
    SELECT SUM("Total Drug Cost") AS total_cost
    FROM medicare
    WHERE Year = 2023
      AND ("Brand Name" ILIKE '%rosuvastatin%' OR "Generic Name" ILIKE '%rosuvastatin%');
- If a drug query returns no rows or NULL, retry with a shorter substring of the term \
(e.g. drop the last few characters) before telling the user data is unavailable.

Specialty matching:
- Use ILIKE '%term%' on Specialty for partial matches.
"""

RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": "Execute a read-only SELECT or WITH query against the `medicare` table and return the rows.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A read-only SQL query (SELECT or WITH only)",
            },
        },
        "required": ["sql"],
    },
}


@st.cache_resource(show_spinner=False)
def _get_chatbot_client() -> anthropic.Anthropic | None:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource(show_spinner=False)
def _get_duckdb_view():
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE VIEW medicare AS SELECT * FROM read_parquet('{_resolved_processed_path()}')"
    )
    return con


_FORBIDDEN_SQL_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "ATTACH", "DETACH", "COPY", "PRAGMA", "EXPORT", "IMPORT",
)


def _execute_sql(sql: str) -> dict:
    sql_clean = sql.strip().rstrip(";").strip()
    if not sql_clean:
        return {"error": "Empty query."}
    upper = sql_clean.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"error": "Only SELECT or WITH queries are permitted."}
    for kw in _FORBIDDEN_SQL_KEYWORDS:
        if f" {kw} " in f" {upper} " or f" {kw}(" in f" {upper}(":
            return {"error": f"The {kw} keyword is not permitted."}
    try:
        result = _get_duckdb_view().execute(sql_clean).fetchdf()
    except Exception as exc:
        return {"error": f"SQL error: {exc}"}

    total_rows = len(result)
    if total_rows == 0:
        return {"row_count": 0, "rows": []}
    truncated = total_rows > CHATBOT_ROW_LIMIT
    preview = result.head(CHATBOT_ROW_LIMIT)
    return {
        "row_count": total_rows,
        "truncated": truncated,
        "rows": preview.astype(object).where(preview.notna(), None).to_dict(orient="records"),
    }


def _is_quota_error(exc: Exception) -> bool:
    return isinstance(exc, anthropic.RateLimitError)


def _run_chatbot_turn(client: anthropic.Anthropic, history: list) -> tuple[str, str | None]:
    """Run the model + tool loop until we get a text answer. Returns (answer, sql_used)."""
    messages = list(history)
    sql_used: str | None = None
    system_blocks = [
        {
            "type": "text",
            "text": CHATBOT_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    for _ in range(CHATBOT_MAX_TOOL_CALLS):
        response = client.messages.create(
            model=CHATBOT_MODEL,
            max_tokens=CHATBOT_MAX_TOKENS,
            system=system_blocks,
            tools=[RUN_SQL_TOOL],
            messages=messages,
        )

        tool_use_block = None
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
            elif block.type == "text":
                text_parts.append(block.text)

        if response.stop_reason != "tool_use" or tool_use_block is None:
            answer = "\n".join(t for t in text_parts if t).strip()
            return (answer or "(no response)", sql_used)

        sql = tool_use_block.input.get("sql", "") if isinstance(tool_use_block.input, dict) else ""
        sql_used = sql
        result = _execute_sql(sql)
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": json.dumps(result, default=str),
                    }
                ],
            }
        )

    return (
        "I wasn't able to land on an answer within my query budget. "
        "Try rephrasing or narrowing the question.",
        sql_used,
    )


def _pair_chat_messages(messages: list[dict]) -> list[tuple[dict, dict | None]]:
    pairs: list[tuple[dict, dict | None]] = []
    i = 0
    while i < len(messages):
        if (
            messages[i]["role"] == "user"
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        ):
            pairs.append((messages[i], messages[i + 1]))
            i += 2
        else:
            pairs.append((messages[i], None))
            i += 1
    return pairs


@st.dialog("Ask AI", width="large")
def _open_chatbot_dialog() -> None:
    render_chatbot()


def _escape_dollars(text: str) -> str:
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def render_chatbot() -> None:
    st.markdown(
        "Ask a question about Medicare Part D prescribing. The bot writes SQL "
        "against the full 2021-2023 dataset (filters from the Dashboard tab don't apply here)."
    )

    client = _get_chatbot_client()
    if client is None:
        st.error(
            "Anthropic API key not configured. Add `ANTHROPIC_API_KEY` to "
            "`.streamlit/secrets.toml` to enable the chatbot."
        )
        return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    suggestion_cols = st.columns(3)
    suggestions = [
        "What were the top 5 drugs by total cost in 2023?",
        "Which state had the highest spending on Eliquis?",
        "How did Cardiology spending change from 2021 to 2023?",
    ]
    pending_input: str | None = None
    for col, suggestion in zip(suggestion_cols, suggestions):
        with col:
            if st.button(suggestion, key=f"chat_suggest_{suggestion}", use_container_width=True):
                pending_input = suggestion

    user_input = st.chat_input("Ask a question about the Medicare Part D data")
    actual_input = user_input or pending_input

    if actual_input:
        st.session_state.chat_messages.append({"role": "user", "content": actual_input})

        history = []
        for msg in st.session_state.chat_messages:
            role = "user" if msg["role"] == "user" else "assistant"
            history.append({"role": role, "content": msg["content"]})

        with st.spinner("Thinking..."):
            try:
                answer, sql_used = _run_chatbot_turn(client, history)
            except Exception as exc:
                err_type = type(exc).__name__
                err_str = str(exc)
                if _is_quota_error(exc):
                    answer = (
                        "Anthropic API rate limit or spend cap reached. "
                        "Please try again later, or check usage in the Anthropic Console.\n\n"
                        f"_Underlying error — **{err_type}**: `{err_str}`_"
                    )
                else:
                    answer = (
                        f"Sorry, something went wrong.\n\n"
                        f"**{err_type}**: `{err_str}`"
                    )
                sql_used = None

        entry = {"role": "assistant", "content": answer}
        if sql_used:
            entry["sql"] = sql_used
        st.session_state.chat_messages.append(entry)

    pairs = _pair_chat_messages(st.session_state.chat_messages)
    if pairs:
        st.markdown("---")
    for idx, (user_msg, assistant_msg) in enumerate(reversed(pairs)):
        if idx > 0:
            st.markdown("---")
        with st.chat_message("user"):
            st.markdown(user_msg["content"])
        if assistant_msg is not None:
            with st.chat_message("assistant"):
                st.markdown(_escape_dollars(assistant_msg["content"]))
                if assistant_msg.get("sql"):
                    with st.expander("Show SQL"):
                        st.code(assistant_msg["sql"], language="sql")


def main() -> None:
    st.markdown(
        """
<div style="
    background: #0a1628;
    padding: 24px 28px 22px;
    border-radius: 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
">
  <div>
    <div style="font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#5b8dd9;margin-bottom:6px;">
      CMS Public Data
    </div>
    <div style="font-size:26px;font-weight:600;color:#f0f4ff;line-height:1.2;">
      Med D Drugs Dashboard
    </div>
    <div style="font-size:13px;color:#7a90b8;margin-top:5px;">
      Interactive analysis of drug costs, claims, and specialty patterns &middot; 2021&ndash;2023
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        df = load_or_build_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    header_extra_cols = st.columns([6, 1])
    with header_extra_cols[1]:
        if st.button(
            "Ask AI",
            icon=":material/auto_awesome:",
            use_container_width=True,
        ):
            _open_chatbot_dialog()

    year_options = sorted(df["Year"].dropna().astype(int).unique().tolist())
    state_options = _select_options(df["State"])
    specialty_options = _select_options(df["Specialty"])

    filter_cols = st.columns([1.1, 1.7, 2.6, 1.3])
    with filter_cols[0]:
        selected_years = st.multiselect("Year", year_options, default=year_options)
    with filter_cols[1]:
        selected_states = st.multiselect("State", state_options)
    with filter_cols[2]:
        selected_specialties = st.multiselect("Specialty", specialty_options)
    with filter_cols[3]:
        grouping = st.radio(
            "Drug grouping",
            ["Brand name", "Generic name"],
            index=0,
            horizontal=True,
        )

    drug_col = "Brand Name" if grouping == "Brand name" else "Generic Name"

    filtered_df = apply_filters(
        df,
        selected_years,
        selected_states,
        selected_specialties,
    )
    specialty_section_df = apply_filters(
        df,
        selected_years,
        selected_states,
        [],
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    context = _filter_context(
        selected_years,
        selected_states,
        selected_specialties,
    )

    render_metric_cards(filtered_df, drug_col)

    st.divider()

    section_heading("Total yearly spending")
    yearly_spending = summarize_yearly_spending(filtered_df)
    st.caption("Total drug cost by year for the current filters.")
    if not yearly_spending.empty:
        years_sorted = yearly_spending["Year"].tolist()
        if len(years_sorted) >= 2:
            first_row = yearly_spending.iloc[0]
            last_row = yearly_spending.iloc[-1]
            first_cost = first_row["Total Drug Cost"]
            last_cost = last_row["Total Drug Cost"]
            if first_cost:
                growth_pct = (last_cost - first_cost) / first_cost * 100
                insight_strip(
                    f"<strong>Total spending trend:</strong> Drug costs changed by "
                    f"<strong>{growth_pct:+.1f}%</strong> from {int(first_row['Year'])} "
                    f"to {int(last_row['Year'])}."
                )
        else:
            only_year = int(yearly_spending.iloc[0]["Year"])
            cost_b = yearly_spending.iloc[0]["Total Drug Cost"] / 1e9
            insight_strip(
                f"<strong>Total spending in {only_year}:</strong> The current filters "
                f"include <strong>${cost_b:.1f}B</strong> in total drug costs."
            )
    yearly_fig = render_yearly_spending_chart(
        yearly_spending,
        _section_title("Total drug cost trend", context=context),
    )
    chart_card(yearly_fig)

    st.divider()

    section_heading("Geographic distribution")
    geographic_section_df = apply_filters(
        df,
        selected_years,
        [],
        selected_specialties,
    )
    geo_context = _filter_context(selected_years, [], selected_specialties)
    st.caption(geo_context)
    map_metric = st.segmented_control(
        "Map metric",
        options=["Total Cost", "Per Capita Cost"],
        default="Total Cost",
        key="map_metric",
    )
    if selected_years:
        population_year = min(max(selected_years), 2023)
    else:
        population_year = 2023
    if map_metric == "Per Capita Cost":
        st.caption(
            f"Drug cost per resident by state (Census {population_year} population). "
            "Hover any state to see total cost, per-capita cost, and population."
        )
    else:
        st.caption(
            "Total drug cost by state across the selected years and specialties. "
            "The state filter is not applied here so all 50 states + DC remain visible."
        )
    if not geographic_section_df.empty:
        valid_state_names = {
            name for name, code in STATE_NAME_TO_CODE.items() if code in USA_MAP_STATE_CODES
        }
        state_totals = (
            geographic_section_df[geographic_section_df["State"].isin(valid_state_names)]
            .groupby("State")["Total Drug Cost"]
            .sum()
            .sort_values(ascending=False)
        )
        if not state_totals.empty:
            if map_metric == "Per Capita Cost":
                pop = load_state_population()
                pop_col = f"Population_{population_year}"
                if pop_col not in pop.columns:
                    pop_col = "Population_2023"
                per_capita = (
                    state_totals.to_frame("Total Drug Cost")
                    .merge(pop[["State", pop_col]], left_index=True, right_on="State")
                    .assign(per_capita=lambda d: d["Total Drug Cost"] / d[pop_col])
                    .sort_values("per_capita", ascending=False)
                )
                if not per_capita.empty:
                    top = per_capita.iloc[0]
                    insight_strip(
                        f"<strong>Highest per-capita state:</strong> {top['State']} at "
                        f"<strong>${top['per_capita']:,.0f}</strong> per resident."
                    )
            else:
                top_state = state_totals.index[0]
                top_state_cost = state_totals.iloc[0]
                insight_strip(
                    f"<strong>Highest spending state:</strong> {top_state} at "
                    f"<strong>{fmt_currency(top_state_cost)}</strong> in total drug costs."
                )
    geo_title = _section_title(
        "Drug cost per capita by state" if map_metric == "Per Capita Cost"
        else "Total drug cost by state",
        context=geo_context,
    )
    geo_fig = render_state_choropleth(
        geographic_section_df,
        geo_title,
        metric=map_metric,
        population_year=population_year,
    )
    chart_card(geo_fig)

    st.divider()

    section_heading("Annual top drugs")
    drug_ctrl_cols = st.columns([3, 1])
    with drug_ctrl_cols[0]:
        top_drug_n = render_top_n_control(
            "Show drugs appearing in each year's top:",
            "top_drug_n",
        )
    with drug_ctrl_cols[1]:
        drug_chart_type = st.segmented_control(
            "Chart type", options=["Bar", "Treemap"], default="Treemap", key="drug_chart_type"
        )
    top_drugs = summarize_top_drugs(filtered_df, grouping, top_drug_n)
    drug_title = _section_title(
        f"Top drugs by total cost",
        grouping=grouping,
        context=context,
    )
    st.caption(context)
    st.caption(
        f"A drug is included if it ranks in the top {top_drug_n} for any selected year. "
        "The chart then shows that drug's full trend across all selected years."
    )
    if not top_drugs.empty:
        pivot = top_drugs.pivot(
            index="Drug Name",
            columns="Year",
            values="Total Drug Cost",
        )
        years = sorted(pivot.columns)
        if len(years) >= 2:
            growth_df = pivot[[years[0], years[-1]]].dropna()
            growth_df = growth_df[growth_df[years[0]] > 0].copy()
            if not growth_df.empty:
                growth_df["growth_pct"] = (
                    (growth_df[years[-1]] - growth_df[years[0]])
                    / growth_df[years[0]]
                    * 100
                )
                top_grower = growth_df["growth_pct"].idxmax()
                growth_val = growth_df.loc[top_grower, "growth_pct"]
                insight_strip(
                    f"<strong>Fastest growing drug:</strong> {top_grower} had the highest cost "
                    f"increase from {years[0]} to {years[-1]} at "
                    f"<strong>{growth_val:+.0f}%</strong>."
                )
    if drug_chart_type == "Treemap":
        drug_fig = render_treemap(filtered_df, drug_col, top_drug_n, drug_title)
    else:
        drug_fig = render_charts(
            top_drugs,
            x="Total Drug Cost",
            y="Drug Name",
            color="Year",
            title=drug_title,
            orientation="h",
        )
    chart_card(drug_fig)
    st.markdown(DATAFRAME_CSS, unsafe_allow_html=True)
    st.dataframe(
        format_tables(
            top_drugs[
                [
                    "Year",
                    "Drug Name",
                    "Total Drug Cost",
                    "Total Claims",
                    "Total 30-Day Fills",
                    "Cost per Claim",
                    "Cost per 30-Day Fill",
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    section_heading("Annual top therapeutic classes")
    ta_ctrl_cols = st.columns([3, 1])
    with ta_ctrl_cols[0]:
        top_ta_n = render_top_n_control(
            "Show therapeutic classes appearing in each year's top:",
            "top_ta_n",
        )
    with ta_ctrl_cols[1]:
        ta_chart_type = st.segmented_control(
            "Chart type", options=["Bar", "Treemap"], default="Treemap", key="ta_chart_type"
        )
    ta_section_df = _attach_atc_level_2(filtered_df)
    top_therapeutic_classes = summarize_top_therapeutic_classes(filtered_df, top_ta_n)
    ta_context = _filter_context(selected_years, selected_states, selected_specialties)
    ta_title = _section_title(
        "Top therapeutic classes (ATC Level 2) by total cost", context=ta_context
    )
    st.caption(ta_context)
    st.caption(
        f"A therapeutic class is included if it ranks in the top {top_ta_n} for any "
        "selected year. Drugs whose generic name isn't yet mapped to an ATC class "
        f"fall under '{OTHERS_ATC_LABEL}'."
    )
    if not top_therapeutic_classes.empty:
        latest_year = top_therapeutic_classes["Year"].max()
        top_ta = (
            top_therapeutic_classes[top_therapeutic_classes["Year"] == latest_year]
            .sort_values("Total Drug Cost", ascending=False)
            .iloc[0]
        )
        cost_b = top_ta["Total Drug Cost"] / 1e9
        insight_strip(
            f"<strong>Highest spending therapeutic class in {int(latest_year)}:</strong> "
            f"{top_ta['Therapeutic Class']} at <strong>${cost_b:.1f}B</strong> in total drug costs."
        )
    if ta_chart_type == "Treemap":
        ta_fig = render_treemap(
            ta_section_df, "Therapeutic Class", top_ta_n, ta_title,
            palette=THERAPEUTIC_TREEMAP_PALETTE,
        )
    else:
        ta_fig = render_charts(
            top_therapeutic_classes,
            x="Total Drug Cost",
            y="Therapeutic Class",
            color="Year",
            title=ta_title,
            orientation="h",
            year_palette=THERAPEUTIC_YEAR_PALETTE,
        )
    chart_card(ta_fig)
    st.dataframe(
        format_tables(
            top_therapeutic_classes[
                [
                    "Year",
                    "Therapeutic Class",
                    "Total Drug Cost",
                    "Total Claims",
                    "Total 30-Day Fills",
                    "Cost per Claim",
                    "Cost per 30-Day Fill",
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    section_heading("Annual top specialties")
    spec_ctrl_cols = st.columns([3, 1])
    with spec_ctrl_cols[0]:
        top_specialty_n = render_top_n_control(
            "Show specialties appearing in each year's top:",
            "top_specialty_n",
        )
    with spec_ctrl_cols[1]:
        specialty_chart_type = st.segmented_control(
            "Chart type", options=["Bar", "Treemap"], default="Treemap", key="specialty_chart_type"
        )
    top_specialties = summarize_top_specialties(specialty_section_df, top_specialty_n)
    specialty_context = _filter_context(selected_years, selected_states, [])
    specialty_title = _section_title("Top specialties by total cost", context=specialty_context)
    st.caption(specialty_context)
    st.caption(
        f"A specialty is included if it ranks in the top {top_specialty_n} for any "
        "selected year. The chart then shows that specialty's full trend across all "
        "selected years."
    )
    if not top_specialties.empty:
        latest_year = top_specialties["Year"].max()
        top_spec = (
            top_specialties[top_specialties["Year"] == latest_year]
            .sort_values("Total Drug Cost", ascending=False)
            .iloc[0]
        )
        cost_b = top_spec["Total Drug Cost"] / 1e9
        insight_strip(
            f"<strong>Highest spending specialty in {int(latest_year)}:</strong> "
            f"{top_spec['Specialty']} at <strong>${cost_b:.1f}B</strong> in total drug costs."
        )
    if specialty_chart_type == "Treemap":
        specialty_fig = render_treemap(
            specialty_section_df, "Specialty", top_specialty_n, specialty_title,
            palette=SPECIALTY_TREEMAP_PALETTE,
        )
    else:
        specialty_fig = render_charts(
            top_specialties,
            x="Total Drug Cost",
            y="Specialty",
            color="Year",
            title=specialty_title,
            orientation="h",
            year_palette=SPECIALTY_YEAR_PALETTE,
        )
    chart_card(specialty_fig)
    st.dataframe(
        format_tables(
            top_specialties[
                [
                    "Year",
                    "Specialty",
                    "Total Drug Cost",
                    "Total Claims",
                    "Total 30-Day Fills",
                    "Cost per Claim",
                    "Cost per 30-Day Fill",
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    section_heading("Yearly drug trend")
    st.markdown(
        "Select up to 5 drugs to compare their total drug cost trend across all selected years."
    )

    trend_options = _select_options(filtered_df[drug_col])
    selected_trend_drugs = st.multiselect(
        f"Select {grouping.lower()} drugs",
        trend_options,
        max_selections=5,
    )

    if selected_trend_drugs:
        trend_df = summarize_trends(filtered_df, grouping, selected_trend_drugs)
        fig = px.line(
            trend_df,
            x="Year",
            y="Total Drug Cost",
            color="Drug Name",
            markers=True,
            template="plotly_white",
            hover_data={
                "Total Drug Cost": ":$,.2f",
                "Total Claims": ":,.0f",
                "Total 30-Day Fills": ":,.0f",
                "Cost per Claim": ":$,.2f",
                "Cost per 30-Day Fill": ":$,.2f",
            },
        )
        fig = style_fig(
            fig,
            title=_section_title(
                "Yearly drug cost trend",
                grouping=grouping,
                context=context,
            ),
        )
        fig.update_layout(legend_title_text="Drug Name")
        tick_vals, tick_text = _build_billions_ticks(float(trend_df["Total Drug Cost"].max()))
        fig.update_xaxes(dtick=1, tickmode="linear")
        fig.update_yaxes(tickvals=tick_vals, ticktext=tick_text, rangemode="tozero")
        chart_card(fig)
        st.dataframe(
            format_tables(
                trend_df[
                    [
                        "Year",
                        "Drug Name",
                        "Total Drug Cost",
                        "Total Claims",
                        "Total 30-Day Fills",
                        "Cost per Claim",
                        "Cost per 30-Day Fill",
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown(
        """
<div style="
    margin-top: 48px;
    padding-top: 16px;
    border-top: 0.5px solid #e8e8e8;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    color: #aaa;
">
  <span>Medicare Part D &middot; CMS public dataset &middot; 2021&ndash;2023</span>
  <span>Data refreshed annually</span>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
