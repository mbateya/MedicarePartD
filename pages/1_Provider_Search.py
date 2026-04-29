from __future__ import annotations

import math

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Provider Search", layout="wide")

HF_DATASET_BASE = (
    "https://huggingface.co/datasets/mbateya/medicare_part_d_prescribers/resolve/main"
)
HF_BASE = f"{HF_DATASET_BASE}/prescribers"
CITIES_URL = f"{HF_DATASET_BASE}/cities.parquet"
RADIUS_OPTIONS = [5, 10, 25, 50, 100]

RADIUS_HELP = (
    "**How the radius works**\n\n"
    "Distance is measured between the center points of cities. A city is "
    "fully included if its center falls within the radius, and fully "
    "excluded otherwise — cities aren't split along their borders.\n\n"
    "**About these cities**\n\n"
    "City names come from each prescriber's listed practice address in the "
    "CMS Medicare Part D data. They may not pinpoint exactly where the "
    "prescriber sees patients, and they are not patient locations."
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


@st.cache_data(show_spinner="Loading cities…", ttl=86400)
def load_cities() -> pd.DataFrame:
    con = _get_con()
    return con.execute(
        f"SELECT State, City, Latitude, Longitude FROM read_parquet('{CITIES_URL}') WHERE Latitude IS NOT NULL"
    ).fetchdf()


@st.cache_data(show_spinner="Querying…", ttl=3600)
def search_radius(
    year: int, state: str, center_city: str, radius_mi: int, drug: str, top_n: int
) -> pd.DataFrame:
    cities = load_cities()
    match = cities[(cities["State"] == state) & (cities["City"] == center_city)]
    if match.empty:
        raise ValueError(f"No coordinates for {center_city}, {state}.")
    lat, lng = float(match.iloc[0]["Latitude"]), float(match.iloc[0]["Longitude"])
    lat_delta = radius_mi / 69.0
    lng_delta = radius_mi / max(1e-3, 69.0 * math.cos(math.radians(lat)))

    con = _get_con()
    url = _state_url(year, state)
    return con.execute(
        f"""
        WITH nearby AS (
            SELECT
                City, Latitude, Longitude,
                3959 * acos(LEAST(1.0,
                    cos(radians(?)) * cos(radians(Latitude))
                      * cos(radians(Longitude) - radians(?))
                    + sin(radians(?)) * sin(radians(Latitude))
                )) AS distance_mi
            FROM read_parquet('{CITIES_URL}')
            WHERE State = ?
              AND Latitude BETWEEN ? AND ?
              AND Longitude BETWEEN ? AND ?
        )
        SELECT
            p."Prescriber Name",
            p."Prescriber NPI",
            p.City,
            ROUND(n.distance_mi, 1) AS "Distance (mi)",
            p.Specialty,
            p."Brand Name",
            p."Generic Name",
            p."Total Claims",
            p."Total Drug Cost"
        FROM read_parquet('{url}') p
        JOIN nearby n USING (City)
        WHERE n.distance_mi <= ?
          AND (p."Brand Name" ILIKE ? OR p."Generic Name" ILIKE ?)
        ORDER BY p."Total Drug Cost" DESC
        LIMIT ?
        """,
        [
            lat, lng, lat,
            state,
            lat - lat_delta, lat + lat_delta,
            lng - lng_delta, lng + lng_delta,
            radius_mi,
            f"%{drug}%", f"%{drug}%",
            top_n,
        ],
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
    if "Distance (mi)" in df.columns:
        fmt["Distance (mi)"] = "{:.1f}"
    return df.style.format(fmt)


st.title("Provider Search")
st.caption(
    "Search Medicare Part D prescribers by city + drug or by provider name. "
    "Data: 2021–2023 CMS Medicare Part D Prescribers, ~78M rows hosted on Hugging Face."
)

mode = st.radio(
    "Search mode",
    [
        "City + Drug → Top Prescribers",
        "City + Radius + Drug → Top Prescribers Within Radius",
        "Provider Name → Top Drugs",
    ],
    horizontal=False,
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

if mode == "City + Drug → Top Prescribers":
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

elif mode == "City + Radius + Drug → Top Prescribers Within Radius":
    cities = load_cities()
    state_cities = sorted(cities[cities["State"] == state]["City"].unique().tolist())
    if not state_cities:
        st.warning(f"No geocoded cities available for {US_STATES[state]}.")
        st.stop()

    q_cols = st.columns([2, 1, 2])
    with q_cols[0]:
        center_city = st.selectbox(
            "Center city",
            options=state_cities,
            index=state_cities.index("Canton") if "Canton" in state_cities else 0,
            help=RADIUS_HELP,
        )
    with q_cols[1]:
        radius_mi = st.selectbox("Radius (mi)", RADIUS_OPTIONS, index=2, help=RADIUS_HELP)
    with q_cols[2]:
        drug = st.text_input("Drug (brand or generic)", placeholder="e.g. Soliris", key="radius_drug")

    st.caption(
        "Same-state radius search. Cities outside the chosen state are not included; "
        "cross-state coverage will come in a later iteration."
    )

    if center_city and drug:
        try:
            df = search_radius(year, state, center_city, radius_mi, drug.strip(), top_n)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()
        if df.empty:
            st.info(
                f"No prescribers of '{drug}' within {radius_mi} mi of {center_city}, {US_STATES[state]} ({year})."
            )
        else:
            st.success(
                f"Top {len(df)} prescribers of {drug} within {radius_mi} mi of {center_city}, {US_STATES[state]} ({year})"
            )
            st.dataframe(_format_currency(df), use_container_width=True, hide_index=True)
    else:
        st.info("Pick a center city, radius, and drug to search.")

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
