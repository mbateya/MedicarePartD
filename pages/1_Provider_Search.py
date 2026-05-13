from __future__ import annotations

import math

import duckdb
import pandas as pd
import streamlit as st

from dashboard_design import render_page_header
from dashboard_tables import render_results_table

HF_DATASET_BASE = (
    "https://huggingface.co/datasets/mbateya/medicare_part_d_prescribers/resolve/main"
)
CITIES_URL = f"{HF_DATASET_BASE}/cities.parquet"
PARTB_DRUG_SPEND_URL = f"{HF_DATASET_BASE}/part_b_drug_spending.parquet"
RADIUS_OPTIONS = [5, 10, 25, 50, 100]

# Med B Provider Search drug-class groupings. Brand-name substrings are
# matched ILIKE against the aggregated Brand Name in `partb_brand_lookup`.
# Searching the group name (e.g. "IVIG") expands the filter to any HCPCS
# whose brand contains one of the listed substrings.
DRUG_GROUPS_PARTB = {
    "IVIG": [
        "Octagam", "Alyglo", "Privigen", "Gammagard", "Gammaked",
        "Bivigam", "Gammaplex", "Carimune", "Panzyga", "Asceniv",
    ],
}

RADIUS_HELP = (
    "**How the radius works**\n\n"
    "Distance is measured between the center points of cities. A city is "
    "fully included if its center falls within the radius, and fully "
    "excluded otherwise — cities aren't split along their borders.\n\n"
    "**About these cities**\n\n"
    "City names come from each prescriber's listed practice address in the "
    "CMS Medicare Part D / Part B data. They may not pinpoint exactly where "
    "the prescriber sees patients, and they are not patient locations."
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

# Per-part schema config. The two CMS datasets use different column names
# for drug identity, cost, and volume — everything else (NPI, City, Specialty,
# partition layout) is shared.
PART_CONFIG = {
    "Part D": {
        "base_url": f"{HF_DATASET_BASE}/prescribers",
        "drug_filter_cols": ['p."Brand Name"', 'p."Generic Name"'],
        "drug_select_cols": ['p."Brand Name"', 'p."Generic Name"'],
        "drug_label": "Drug (brand or generic)",
        "drug_placeholder": "e.g. Soliris",
        "cost_col": 'p."Total Drug Cost"',
        "cost_alias": "Total Drug Cost",
        "volume_col": 'p."Total Claims"',
        "volume_alias": "Total Claims",
        "per_unit_alias": "Cost per 30-Day Fill",
        "per_unit_expr": 'p."Total Drug Cost" / NULLIF(p."Total 30-Day Fills", 0)',
        "extra_provider_cols": ['p."Total Beneficiaries"'],
        "brand_join": "",
        "provider_alias": "Prescriber Name",   # Part D records WHO PRESCRIBED the pharmacy script
        "npi_alias": "Prescriber NPI",
        "provider_term": "prescribers",
    },
    "Part B": {
        "base_url": f"{HF_DATASET_BASE}/partb_prescribers",
        # Brand & Generic come from a runtime LEFT JOIN against
        # part_b_drug_spending.parquet (registered as `partb_brand_lookup`
        # view on the cached DuckDB connection).
        "drug_filter_cols": [
            'p."HCPCS Code"', 'p."HCPCS Description"',
            'b."Brand Name"', 'b."Generic Name"', 'b."Drug Group"',
        ],
        "drug_select_cols": [
            'p."HCPCS Code"', 'p."HCPCS Description"',
            'b."Brand Name"', 'b."Generic Name"',
        ],
        "drug_label": "Drug (brand, generic, or HCPCS code/description, e.g. Keytruda)",
        "drug_placeholder": "e.g. Keytruda",
        "cost_col": 'p."Total Spending"',
        "cost_alias": "Total Spending",
        "volume_col": 'p."Total Services"',
        "volume_alias": "Total Services",
        "per_unit_alias": "Avg Medicare Payment",
        "per_unit_expr": 'p."Avg Medicare Payment"',
        "extra_provider_cols": ['p."Total Beneficiaries"'],
        "brand_join": 'LEFT JOIN partb_brand_lookup b USING ("HCPCS Code")',
        # Part B PUF records the RENDERING provider (who administered the drug
        # and billed Medicare), not necessarily the ordering clinician.
        "provider_alias": "Rendering Provider",
        "npi_alias": "Rendering NPI",
        "provider_term": "rendering providers",
    },
}


def _build_drug_group_case() -> str:
    """SQL CASE expression mapping aggregated Brand Name → drug class group."""
    if not DRUG_GROUPS_PARTB:
        return "CAST(NULL AS VARCHAR)"
    parts = []
    for group, brands in DRUG_GROUPS_PARTB.items():
        conds = " OR ".join(f"\"Brand Name\" ILIKE '%{b}%'" for b in brands)
        parts.append(f"WHEN {conds} THEN '{group}'")
    return "CASE " + " ".join(parts) + " END"


@st.cache_resource(show_spinner="Connecting to dataset…")
def _get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Persistent HCPCS Code → Brand Name / Generic Name / Drug Group lookup
    # for Part B provider search. STRING_AGG handles the rare biosimilar case
    # where one HCPCS maps to multiple brand names. The Drug Group column
    # tags class-level groupings (e.g. IVIG) so users can search by class.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW partb_brand_lookup AS
        WITH agg AS (
            SELECT
                "HCPCS Code",
                STRING_AGG(DISTINCT "Brand Name",   '; ' ORDER BY "Brand Name")   AS "Brand Name",
                STRING_AGG(DISTINCT "Generic Name", '; ' ORDER BY "Generic Name") AS "Generic Name"
            FROM read_parquet('{PARTB_DRUG_SPEND_URL}')
            WHERE "HCPCS Code" IS NOT NULL
            GROUP BY "HCPCS Code"
        )
        SELECT
            "HCPCS Code", "Brand Name", "Generic Name",
            {_build_drug_group_case()} AS "Drug Group"
        FROM agg
        """
    )
    return con


def _state_url(part: str, year: int, state: str) -> str:
    return f"{PART_CONFIG[part]['base_url']}/year={year}/State={state}/data_0.parquet"


@st.cache_data(show_spinner="Loading cities…", ttl=86400)
def load_cities() -> pd.DataFrame:
    con = _get_con()
    return con.execute(
        f"SELECT State, City, Latitude, Longitude FROM read_parquet('{CITIES_URL}') WHERE Latitude IS NOT NULL"
    ).fetchdf()


@st.cache_data(show_spinner="Querying…", ttl=3600)
def search_radius(
    part: str, year: int, state: str, center_city: str, radius_mi: int, drug: str, top_n: int
) -> pd.DataFrame:
    cfg = PART_CONFIG[part]
    cities = load_cities()
    match = cities[(cities["State"] == state) & (cities["City"] == center_city)]
    if match.empty:
        raise ValueError(f"No coordinates for {center_city}, {state}.")
    lat, lng = float(match.iloc[0]["Latitude"]), float(match.iloc[0]["Longitude"])
    lat_delta = radius_mi / 69.0
    lng_delta = radius_mi / max(1e-3, 69.0 * math.cos(math.radians(lat)))

    con = _get_con()
    url = _state_url(part, year, state)
    drug_select = ", ".join(cfg["drug_select_cols"])
    drug_filter = " OR ".join(f"{c} ILIKE ?" for c in cfg["drug_filter_cols"])
    drug_filter_params = [f"%{drug}%"] * len(cfg["drug_filter_cols"])
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
            p."Prescriber Name" AS "{cfg["provider_alias"]}",
            p."Prescriber NPI"  AS "{cfg["npi_alias"]}",
            p.City,
            ROUND(n.distance_mi, 1) AS "Distance (mi)",
            p.Specialty,
            {drug_select},
            {cfg["volume_col"]} AS "{cfg["volume_alias"]}",
            {cfg["cost_col"]} AS "{cfg["cost_alias"]}"
        FROM read_parquet('{url}') p
        {cfg["brand_join"]}
        JOIN nearby n USING (City)
        WHERE n.distance_mi <= ?
          AND ({drug_filter})
        ORDER BY {cfg["cost_col"]} DESC
        LIMIT ?
        """,
        [
            lat, lng, lat,
            state,
            lat - lat_delta, lat + lat_delta,
            lng - lng_delta, lng + lng_delta,
            radius_mi,
            *drug_filter_params,
            top_n,
        ],
    ).fetchdf()


@st.cache_data(show_spinner="Querying…", ttl=3600)
def search_provider_drugs(part: str, year: int, state: str, name: str, top_n: int) -> pd.DataFrame:
    cfg = PART_CONFIG[part]
    con = _get_con()
    url = _state_url(part, year, state)
    drug_select = ", ".join(cfg["drug_select_cols"])
    extra = ", ".join(cfg["extra_provider_cols"])
    return con.execute(
        f"""
        SELECT
            p."Prescriber Name" AS "{cfg["provider_alias"]}",
            p.City,
            p.Specialty,
            {drug_select},
            {cfg["volume_col"]} AS "{cfg["volume_alias"]}",
            {cfg["cost_col"]} AS "{cfg["cost_alias"]}",
            {extra}
        FROM read_parquet('{url}') p
        {cfg["brand_join"]}
        WHERE p."Prescriber Name" ILIKE ?
        ORDER BY {cfg["cost_col"]} DESC
        LIMIT ?
        """,
        [f"%{name}%", top_n],
    ).fetchdf()


render_page_header(
    title="Provider Search",
    subtitle=(
        "Search Medicare Part D prescribers or Part B rendering providers by city, "
        "radius, drug, or provider name. Ask AI can answer aggregate rollup questions."
    ),
    section="Search / Providers",
    icon="⌕",
)

part = st.radio(
    "Medicare Part",
    ["Part D", "Part B"],
    horizontal=True,
    help=(
        "**Part D** = pharmacy-dispensed drugs; data records the **prescriber** "
        "(MD who wrote the script). ~78M prescriber rows.\n\n"
        "**Part B** = clinician-administered drugs (infusions, injectables); data "
        "records the **rendering provider** (who administered the drug and billed "
        "Medicare). The rendering provider is usually but not always the ordering "
        "clinician. Search by drug brand, generic, or HCPCS."
    ),
)

mode = st.radio(
    "Search mode",
    [
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

cfg = PART_CONFIG[part]

if mode == "City + Radius + Drug → Top Prescribers Within Radius":
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
        drug = st.text_input(cfg["drug_label"], placeholder=cfg["drug_placeholder"], key="radius_drug")

    st.caption(
        "Same-state radius search. Cities outside the chosen state are not included; "
        "cross-state coverage will come in a later iteration."
    )

    if center_city and drug:
        try:
            df = search_radius(part, year, state, center_city, radius_mi, drug.strip(), top_n)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()
        if df.empty:
            st.info(
                f"No {part} {cfg['provider_term']} of '{drug}' within {radius_mi} mi of {center_city}, {US_STATES[state]} ({year})."
            )
        else:
            st.success(
                f"Top {len(df)} {part} {cfg['provider_term']} of {drug} within {radius_mi} mi of {center_city}, {US_STATES[state]} ({year})"
            )
            render_results_table(
                df,
                primary_metric=cfg["cost_alias"],
                entity_col=cfg["provider_alias"],
            )
    else:
        st.info("Pick a center city, radius, and drug to search.")

else:
    name = st.text_input("Provider name", placeholder="e.g. Smith")
    if name:
        try:
            df = search_provider_drugs(part, year, state, name.strip(), top_n)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()
        if df.empty:
            st.info(f"No {part} providers matching '{name}' found in {US_STATES[state]} {year}.")
        else:
            providers = df[cfg["provider_alias"]].nunique()
            label = (
                f"Top {len(df)} {part} drugs by spend for '{name}' in {US_STATES[state]} {year}"
                if providers == 1
                else f"Top {len(df)} (provider, drug) pairs matching '{name}' in {US_STATES[state]} {year}"
            )
            st.success(label)
            render_results_table(
                df,
                primary_metric=cfg["cost_alias"],
                entity_col=cfg["provider_alias"],
            )
    else:
        st.info("Enter a provider name to search.")
