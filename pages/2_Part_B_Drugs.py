from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download

HF_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_PART_B_FILE = "part_b_drug_spending.parquet"
TOP_N_OPTIONS = [5, 10, 20]

PALETTE_BARS = ["#9b6dde", "#c4a3eb", "#7a3fd1"]   # purple family, distinct from Part D
PALETTE_TREEMAP = ["#5a2ea8", "#7a3fd1", "#9b6dde", "#b993e8", "#c4a3eb", "#d8c5f0", "#e8dbf7"]


@st.cache_data(show_spinner="Loading Part B drug spending…", ttl=86400)
def load_part_b() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_PART_B_FILE,
        repo_type="dataset",
    )
    return pd.read_parquet(path)


def _fmt_currency(val: float) -> str:
    for divisor, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if val >= divisor:
            s = f"{val / divisor:.2f}".rstrip("0").rstrip(".")
            return f"${s}{suffix}"
    return f"${val:,.0f}"


df_full = load_part_b()
years_available = sorted(df_full["Year"].dropna().unique().astype(int).tolist())
generics_available = sorted(df_full["Generic Name"].dropna().unique().tolist())

st.title("Medicare Part B Drug Spending")
st.caption(
    "Annual Medicare Part B drug spending from CMS public data (one row per "
    "drug per year). Part B covers drugs administered by clinicians "
    "(infusions, injectables in office/clinic) — distinct from the Part D "
    "pharmacy-dispensed drugs on the Dashboard tab."
)

filter_cols = st.columns([2, 3, 1])
with filter_cols[0]:
    selected_years = st.multiselect(
        "Years",
        years_available,
        default=years_available,
    )
with filter_cols[1]:
    selected_generics = st.multiselect(
        "Generic name (optional filter)",
        generics_available,
        placeholder="Type to filter to specific drugs…",
    )
with filter_cols[2]:
    top_n = st.selectbox("Top N", TOP_N_OPTIONS, index=1)

if not selected_years:
    st.warning("Select at least one year to see results.")
    st.stop()

filtered = df_full[df_full["Year"].isin(selected_years)].copy()
if selected_generics:
    filtered = filtered[filtered["Generic Name"].isin(selected_generics)]

# KPI strip
total_spend = filtered["Total Spending"].sum()
total_claims = filtered["Total Claims"].sum()
total_benes = filtered["Total Beneficiaries"].sum()
n_drugs = filtered["Generic Name"].nunique()

kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Part B drug spend", _fmt_currency(total_spend))
kpi_cols[1].metric("Total claims", f"{int(total_claims):,}")
kpi_cols[2].metric("Total beneficiaries", f"{int(total_benes):,}")
kpi_cols[3].metric("Distinct drugs (generic)", f"{n_drugs:,}")

st.divider()

st.subheader(f"Top {top_n} drugs by total Part B spend")
chart_type = st.segmented_control(
    "Chart type", options=["Bar", "Treemap"], default="Bar", key="ptb_chart_type",
)
top_generics = (
    filtered.groupby("Generic Name", as_index=False)["Total Spending"].sum()
    .sort_values("Total Spending", ascending=False)
    .head(top_n)
)
top_names = top_generics["Generic Name"].tolist()
top_with_year = (
    filtered[filtered["Generic Name"].isin(top_names)]
    .groupby(["Year", "Generic Name"], as_index=False)["Total Spending"].sum()
)
top_with_year["Year"] = top_with_year["Year"].astype(str)
ordered_years = sorted(top_with_year["Year"].unique())

if chart_type == "Bar":
    color_map = {y: PALETTE_BARS[i % len(PALETTE_BARS)] for i, y in enumerate(ordered_years)}
    fig = px.bar(
        top_with_year,
        x="Total Spending",
        y="Generic Name",
        color="Year",
        category_orders={"Generic Name": top_names, "Year": ordered_years},
        orientation="h",
        template="plotly_white",
        color_discrete_map=color_map,
        barmode="group",
    )
    fig.update_layout(
        height=max(380, 38 * len(top_names)),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Total Spending (USD)",
        margin=dict(t=20, l=10, r=10, b=10),
    )
else:
    treemap_df = top_generics.copy()
    others = filtered[~filtered["Generic Name"].isin(top_names)]["Total Spending"].sum()
    if others > 0:
        treemap_df = pd.concat(
            [treemap_df, pd.DataFrame([{"Generic Name": f"Others ({n_drugs - len(top_names)} more)", "Total Spending": others}])],
            ignore_index=True,
        )
    fig = px.treemap(
        treemap_df,
        path=["Generic Name"],
        values="Total Spending",
        color="Generic Name",
        color_discrete_sequence=PALETTE_TREEMAP + ["#cccccc"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:$,.0f}",
        hovertemplate="<b>%{label}</b><br>Total Spending: %{value:$,.0f}<extra></extra>",
    )
    fig.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=520)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Yearly trend for selected drugs")
trend_options = top_names if not selected_generics else selected_generics
trend_picks = st.multiselect(
    "Drugs to plot (defaults to current top list)",
    sorted(set(trend_options)),
    default=trend_options[:5],
    max_selections=8,
)
if trend_picks:
    trend_df = (
        filtered[filtered["Generic Name"].isin(trend_picks)]
        .groupby(["Year", "Generic Name"], as_index=False)["Total Spending"].sum()
    )
    trend_fig = px.line(
        trend_df.sort_values(["Generic Name", "Year"]),
        x="Year",
        y="Total Spending",
        color="Generic Name",
        markers=True,
        template="plotly_white",
    )
    trend_fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis_title="Total Spending (USD)",
        margin=dict(t=20, l=10, r=10, b=10),
        height=420,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

st.divider()

st.subheader("Drill-down: full detail by HCPCS code")
st.caption(
    "One row per (HCPCS code, brand, year). HCPCS codes are how Part B drugs are billed; "
    "the same generic may have multiple HCPCS codes (e.g. different doses, formulations)."
)
display = filtered.sort_values(["Year", "Total Spending"], ascending=[True, False]).copy()
display_cols = [
    "Year", "HCPCS Code", "HCPCS Description", "Brand Name", "Generic Name",
    "Total Spending", "Total Dosage Units", "Total Claims", "Total Beneficiaries",
    "Avg Spending per Dosage Unit", "Avg Spending per Beneficiary",
]
display = display[[c for c in display_cols if c in display.columns]]
fmt = {
    "Total Spending": "${:,.0f}",
    "Total Dosage Units": "{:,.0f}",
    "Total Claims": "{:,.0f}",
    "Total Beneficiaries": "{:,.0f}",
    "Avg Spending per Dosage Unit": "${:,.2f}",
    "Avg Spending per Beneficiary": "${:,.2f}",
}
fmt = {k: v for k, v in fmt.items() if k in display.columns}
st.dataframe(display.style.format(fmt), use_container_width=True, hide_index=True)
