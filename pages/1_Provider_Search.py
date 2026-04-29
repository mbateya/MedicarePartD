from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Provider Search", layout="wide")

HF_BASE = (
    "https://huggingface.co/datasets/mbateya/medicare_part_d_prescribers/"
    "resolve/main/prescribers"
)

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam", "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}
STATE_OPTIONS = sorted(US_STATES.items(), key=lambda kv: kv[1])
YEAR_OPTIONS = [2023, 2022, 2021]
TOP_N_OPTIONS = [5, 10, 20]


@st.cache_resource(show_spinner="Connecting to dataset…")
def _get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def _state_url(year: int, state: str) -> str:
    return f"{HF_BASE}/year={year}/State={state}/data_0.parquet"


@st.cache_data(show_spinner="Querying…", ttl=3600)
def search_city_drug(year: int, state: str, city: str, drug: str, top_n: int) -> pd.DataFrame:
    con = _get_con()
    url = _state_url(year, state)
    return con.execute(
        f"""
        SELECT
            "Prescriber Name",
            "Prescriber NPI",
            City,
            Specialty,
            "Brand Name",
            "Generic Name",
            "Total Claims",
            "Total Drug Cost",
            "Cost per 30-Day Fill",
        FROM (
            SELECT
                *,
                "Total Drug Cost" / NULLIF("Total 30-Day Fills", 0) AS "Cost per 30-Day Fill"
            FROM read_parquet('{url}')
            WHERE City ILIKE ?
              AND ("Brand Name" ILIKE ? OR "Generic Name" ILIKE ?)
        )
        ORDER BY "Total Drug Cost" DESC
        LIMIT ?
        """,
        [f"%{city}%", f"%{drug}%", f"%{drug}%", top_n],
    ).fetchdf()


@st.cache_data(show_spinner="Querying…", ttl=3600)
def search_provider_drugs(year: int, state: str, name: str, top_n: int) -> pd.DataFrame:
    con = _get_con()
    url = _state_url(year, state)
    return con.execute(
        f"""
        SELECT
            "Prescriber Name",
            City,
            Specialty,
            "Brand Name",
            "Generic Name",
            "Total Claims",
            "Total Drug Cost",
            "Total Beneficiaries"
        FROM read_parquet('{url}')
        WHERE "Prescriber Name" ILIKE ?
        ORDER BY "Total Drug Cost" DESC
        LIMIT ?
        """,
        [f"%{name}%", top_n],
    ).fetchdf()


def _format_currency(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    money_cols = [c for c in ("Total Drug Cost", "Cost per 30-Day Fill") if c in df.columns]
    fmt = {c: "${:,.2f}" for c in money_cols}
    if "Total Claims" in df.columns:
        fmt["Total Claims"] = "{:,.0f}"
    if "Total Beneficiaries" in df.columns:
        fmt["Total Beneficiaries"] = "{:,.0f}"
    return df.style.format(fmt)


st.title("Provider Search")
st.caption(
    "Search Medicare Part D prescribers by city + drug or by provider name. "
    "Data: 2021–2023 CMS Medicare Part D Prescribers, ~78M rows hosted on Hugging Face."
)

mode = st.radio(
    "Search mode",
    ["City + Drug → Top Prescribers", "Provider Name → Top Drugs"],
    horizontal=True,
)

control_cols = st.columns([1, 2, 1])
with control_cols[0]:
    year = st.selectbox("Year", YEAR_OPTIONS, index=0)
with control_cols[1]:
    state = st.selectbox(
        "State",
        options=[abbr for abbr, _ in STATE_OPTIONS],
        format_func=lambda a: f"{US_STATES[a]} ({a})",
        index=[abbr for abbr, _ in STATE_OPTIONS].index("MI"),
    )
with control_cols[2]:
    top_n = st.selectbox("Top N", TOP_N_OPTIONS, index=1)

st.divider()

if mode.startswith("City"):
    q_cols = st.columns(2)
    with q_cols[0]:
        city = st.text_input("City", placeholder="e.g. Ann Arbor")
    with q_cols[1]:
        drug = st.text_input("Drug (brand or generic)", placeholder="e.g. Soliris")

    if city and drug:
        try:
            df = search_city_drug(year, state, city.strip(), drug.strip(), top_n)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()
        if df.empty:
            st.info(f"No prescribers of '{drug}' found in cities matching '{city}' in {US_STATES[state]} {year}.")
        else:
            st.success(f"Top {len(df)} prescribers of {drug} in cities matching '{city}', {US_STATES[state]} {year}")
            st.dataframe(_format_currency(df), use_container_width=True, hide_index=True)
    else:
        st.info("Enter a city and a drug name to search.")

else:
    name = st.text_input("Provider name", placeholder="e.g. Smith")
    if name:
        try:
            df = search_provider_drugs(year, state, name.strip(), top_n)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()
        if df.empty:
            st.info(f"No providers matching '{name}' found in {US_STATES[state]} {year}.")
        else:
            providers = df["Prescriber Name"].nunique()
            label = (
                f"Top {len(df)} drugs by spend for '{name}' in {US_STATES[state]} {year}"
                if providers == 1
                else f"Top {len(df)} (provider, drug) pairs matching '{name}' in {US_STATES[state]} {year}"
            )
            st.success(label)
            st.dataframe(_format_currency(df), use_container_width=True, hide_index=True)
    else:
        st.info("Enter a provider name to search.")
