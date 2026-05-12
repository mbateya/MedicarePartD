from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download

from dashboard_tables import render_detail_table

HF_DATASET_ID = "mbateya/medicare_part_d_prescribers"
HF_PART_B_FILE = "part_b_drug_spending.parquet"
HF_PARTB_SPECIALTY_FILE = "partb_drug_by_specialty.parquet"

# Multi-hue palettes for high inter-category contrast.
# Header/section accents stay purple; data marks use distinct hues for readability.
PALETTE_YEARS = ["#5a2ea8", "#1d9e75", "#ef9f27", "#185fa5", "#d85a30"]
PALETTE_DRUGS = [
    "#5a2ea8", "#1d9e75", "#ef9f27", "#185fa5", "#d85a30",
    "#c0392b", "#16a085", "#9b6dde", "#3aae8e", "#f5c178",
    "#0c447c", "#a44a3f",
]
HEADER_BG = "#1c2a5e"      # deep indigo — distinct from Part D's #0a1628
HEADER_EYEBROW = "#9b8fd9"
HEADER_SUBTITLE = "#a8b3d9"
HEADER_TITLE = "#f0eefa"
ACCENT = "#7a3fd1"


@st.cache_data(show_spinner="Loading Part B drug spending…", ttl=86400)
def load_part_b() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_PART_B_FILE,
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    # CMS appends '*' to some brand names as a footnote marker; strip for clean display
    df["Brand Name"] = df["Brand Name"].astype(str).str.rstrip("*").str.strip()
    return df


@st.cache_data(show_spinner="Loading Part B specialty rollup…", ttl=86400)
def load_partb_specialty() -> pd.DataFrame:
    """Year × Specialty × HCPCS rollup built from the CMS Physician PUF (drug HCPCS only)."""
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_PARTB_SPECIALTY_FILE,
        repo_type="dataset",
    )
    return pd.read_parquet(path)


def _fmt_num(value: float, unit: float) -> str:
    return f"{value / unit:.1f}".rstrip("0").rstrip(".")


def _fmt_cost(value: float) -> str:
    if value >= 1e9:
        return f"${_fmt_num(value, 1e9)}B"
    if value >= 1e6:
        return f"${_fmt_num(value, 1e6)}M"
    if value >= 1e3:
        return f"${_fmt_num(value, 1e3)}K"
    return f"${value:,.0f}"


def _fmt_count(value: float) -> str:
    if value >= 1e9:
        return f"{_fmt_num(value, 1e9)}B"
    if value >= 1e6:
        return f"{_fmt_num(value, 1e6)}M"
    if value >= 1e3:
        return f"{_fmt_num(value, 1e3)}K"
    return f"{value:,.0f}"


def compute_others_stats(
    df: pd.DataFrame,
    name_col: str,
    top_n: int,
    value_col: str = "Total Spending",
) -> dict:
    """Aggregate the 'Others' tail beyond top N for sidebar display."""
    totals = (
        df.groupby(name_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
    )
    grand_total = totals.sum()
    if len(totals) <= top_n or grand_total <= 0:
        return {"count": 0, "value": 0.0, "pct": 0.0}
    others = totals.iloc[top_n:]
    return {
        "count": len(others),
        "value": float(others.sum()),
        "pct": float(others.sum() / grand_total * 100),
    }


def render_others_card(
    stats: dict,
    top_n: int,
    label_singular: str = "drug",
    label_plural: str | None = None,
    accent: str = "#888",
) -> None:
    """Compact sidebar card summarising the 'Others' tail next to a treemap."""
    if stats["count"] == 0:
        return
    if label_plural is None:
        label_plural = f"{label_singular}s"
    label = label_singular if stats["count"] == 1 else label_plural
    st.markdown(
        f"""
<div style="
    background:white;
    border:0.5px solid #e8e8e8;
    border-left:3px solid {accent};
    border-radius:10px;
    padding:18px 16px;
    margin:0 0 12px 0;
">
  <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:10px;">
    Beyond top {top_n}
  </div>
  <div style="font-size:22px;font-weight:600;color:#111;line-height:1.1;margin-bottom:4px;">
    {_fmt_cost(stats['value'])}
  </div>
  <div style="font-size:13px;color:#444;margin-bottom:12px;">
    across {stats['count']:,} more {label}
  </div>
  <div style="font-size:13px;color:#888;">
    {stats['pct']:.0f}% of all spend
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _currency_axis_ticks(max_val: float) -> tuple[list[float], list[str]]:
    """Return (tickvals, ticktext) for a currency axis using $XB / $XM abbreviations."""
    import math
    if max_val <= 0:
        return [0], ["$0"]
    raw_step = max_val / 5
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 5):
        if m * magnitude >= raw_step:
            step = m * magnitude
            break
    else:
        step = 10 * magnitude
    n_ticks = int(max_val / step) + 2
    tickvals = [i * step for i in range(n_ticks)]
    ticktext = [_fmt_cost(v) if v > 0 else "$0" for v in tickvals]
    return tickvals, ticktext


def section_heading(text: str) -> None:
    st.markdown(
        f"""
<div style="margin: 28px 0 4px;">
  <span style="font-size:18px;font-weight:600;color:{HEADER_BG};">{text}</span>
  <div style="height:2px;width:32px;background:{ACCENT};border-radius:1px;margin-top:5px;"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def insight_strip(text: str) -> None:
    st.markdown(
        f"""
<div style="
    background:#f6f2ff;
    border-left:3px solid {ACCENT};
    border-radius:0 8px 8px 0;
    padding:10px 14px;
    font-size:13px;
    color:{HEADER_BG};
    margin:8px 0 14px;
    line-height:1.6;
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


def render_top_n_control(label: str, key: str) -> int:
    selection = st.segmented_control(
        label,
        options=[5, 10, 20],
        default=10,
        key=key,
    )
    return int(selection or 10)


def render_year_control(year_options: list[int], key: str) -> list[int]:
    options = ["All", *year_options]
    selection = st.segmented_control(
        "Years",
        options=options,
        default="All",
        key=key,
    )
    if selection == "All" or selection is None:
        return year_options
    return [int(selection)]


def render_metric_cards(filtered: pd.DataFrame, drug_col: str) -> None:
    total_spend = filtered["Total Spending"].sum()
    total_claims = filtered["Total Claims"].sum()
    n_drugs = filtered[drug_col].nunique()
    drug_label = "By brand" if drug_col == "Brand Name" else "By generic name"

    years_sorted = sorted(filtered["Year"].dropna().astype(int).unique())
    if len(years_sorted) >= 2:
        first = filtered[filtered["Year"] == years_sorted[0]]["Total Spending"].sum()
        last = filtered[filtered["Year"] == years_sorted[-1]]["Total Spending"].sum()
        growth_pct = (last - first) / first * 100 if first else 0
        growth_str = f"{growth_pct:+.1f}%"
        growth_sub = f"{_fmt_cost(first)} &rarr; {_fmt_cost(last)}"
    else:
        growth_str = "N/A"
        growth_sub = "Select 2+ years"

    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:14px 0 4px;">

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:8px;padding:10px 12px;position:relative;overflow:hidden;min-height:82px;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#7a3fd1;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#888;margin-bottom:5px;">Total Part B spend</div>
    <div style="font-size:21px;font-weight:600;color:#111;line-height:1;">{_fmt_cost(total_spend)}</div>
    <div style="font-size:11px;color:#1d9e75;margin-top:4px;">Selected period</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:8px;padding:10px 12px;position:relative;overflow:hidden;min-height:82px;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#9b6dde;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#888;margin-bottom:5px;">Total claims</div>
    <div style="font-size:21px;font-weight:600;color:#111;line-height:1;">{_fmt_count(total_claims)}</div>
    <div style="font-size:11px;color:#888;margin-top:4px;">Selected period</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:8px;padding:10px 12px;position:relative;overflow:hidden;min-height:82px;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#d85a30;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#888;margin-bottom:5px;">Distinct drugs</div>
    <div style="font-size:21px;font-weight:600;color:#111;line-height:1;">{n_drugs:,}</div>
    <div style="font-size:11px;color:#888;margin-top:4px;">{drug_label}</div>
  </div>

  <div style="background:white;border:0.5px solid #e8e8e8;border-radius:8px;padding:10px 12px;position:relative;overflow:hidden;min-height:82px;">
    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#5a2ea8;border-radius:10px 0 0 10px;"></div>
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#888;margin-bottom:5px;">Spend growth</div>
    <div style="font-size:21px;font-weight:600;color:#111;line-height:1;">{growth_str}</div>
    <div style="font-size:11px;color:#888;margin-top:4px;">{growth_sub}</div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )


def summarize_yearly_spending(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "Total Spending": "sum",
        "Total Claims": "sum",
    }
    if "Total Dosage Units" in df.columns:
        agg_cols["Total Dosage Units"] = "sum"
    return (
        df.groupby("Year", as_index=False)
        .agg(agg_cols)
        .sort_values("Year")
        .reset_index(drop=True)
    )


def render_yearly_spending_chart(df: pd.DataFrame):
    chart_df = df.copy()
    chart_df["_spend"] = chart_df["Total Spending"].apply(_fmt_cost)
    chart_df["_claims"] = chart_df["Total Claims"].apply(_fmt_count)
    custom_cols = ["_spend", "_claims"]
    hover_lines = "<br>Total Claims: %{customdata[1]}"
    if "Total Dosage Units" in chart_df.columns:
        chart_df["_units"] = chart_df["Total Dosage Units"].apply(_fmt_count)
        custom_cols.append("_units")
        hover_lines += "<br>Total Dosage Units: %{customdata[2]}"

    fig = px.area(
        chart_df,
        x="Year",
        y="Total Spending",
        template="plotly_white",
        custom_data=custom_cols,
    )
    fig.update_traces(
        mode="lines+markers",
        line=dict(color=ACCENT, width=3),
        marker=dict(size=9, color=ACCENT),
        fillcolor="rgba(155, 109, 222, 0.22)",
        hovertemplate="<b>%{x}</b><br>Total Spending: %{customdata[0]}"
        + hover_lines
        + "<extra></extra>",
    )

    y_min = float(chart_df["Total Spending"].min())
    y_max = float(chart_df["Total Spending"].max())
    if y_min == y_max:
        y_padding = y_max * 0.05 if y_max else 1
        y_range = [max(0, y_min - y_padding), y_max + y_padding]
    else:
        y_range = [y_min * 0.95, y_max * 1.05]

    for index, row in chart_df.reset_index(drop=True).iterrows():
        if index == 0:
            continue
        prev = chart_df.iloc[index - 1]["Total Spending"]
        curr = row["Total Spending"]
        if prev:
            pct = (curr - prev) / prev * 100
            fig.add_annotation(
                x=row["Year"],
                y=curr,
                text=f"{pct:+.1f}%",
                showarrow=False,
                yshift=14,
                font=dict(size=12, color=ACCENT),
            )

    tickvals, ticktext = _currency_axis_ticks(float(chart_df["Total Spending"].max()))
    fig.update_layout(
        showlegend=False,
        height=360,
        margin=dict(t=20, l=10, r=10, b=10),
        xaxis=dict(dtick=1, tickmode="linear", title="Year"),
        yaxis=dict(
            title="Total Spending",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=y_range,
        ),
    )
    return fig


df_full = load_part_b()
years_available = sorted(df_full["Year"].dropna().unique().astype(int).tolist())

st.markdown(
    f"""
<div style="
    background: {HEADER_BG};
    padding: 24px 28px 22px;
    border-radius: 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
">
  <div>
    <div style="font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:{HEADER_EYEBROW};margin-bottom:6px;">
      CMS Public Data
    </div>
    <div style="font-size:26px;font-weight:600;color:{HEADER_TITLE};line-height:1.2;">
      Med B Drugs Dashboard
    </div>
    <div style="font-size:13px;color:{HEADER_SUBTITLE};margin-top:5px;">
      Clinician-administered drugs (infusions, injectables) billed by HCPCS code &middot; 2019&ndash;2023
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Annual Medicare Part B drug spending from CMS public data (one row per "
    "drug per year). Part B covers drugs administered by clinicians "
    "(infusions, injectables in office/clinic) — distinct from the Part D "
    "pharmacy-dispensed drugs on the Dashboard tab."
)

filter_cols = st.columns([1.6, 1.3])
with filter_cols[0]:
    selected_years = render_year_control(years_available, "ptb_year")
with filter_cols[1]:
    grouping = st.radio(
        "Drug grouping",
        ["Brand name", "Generic name"],
        index=0,
        horizontal=True,
    )

drug_col = "Brand Name" if grouping == "Brand name" else "Generic Name"

if not selected_years:
    st.warning("Select at least one year to see results.")
    st.stop()

filtered = df_full[df_full["Year"].isin(selected_years)].copy()

render_metric_cards(filtered, drug_col)

st.divider()

section_heading("Total yearly spending")
yearly_spending = summarize_yearly_spending(filtered)
st.caption("Total Part B drug spending by year for the current filters.")
if not yearly_spending.empty:
    if len(yearly_spending) >= 2:
        first_row = yearly_spending.iloc[0]
        last_row = yearly_spending.iloc[-1]
        first_spend = first_row["Total Spending"]
        last_spend = last_row["Total Spending"]
        if first_spend:
            growth_pct = (last_spend - first_spend) / first_spend * 100
            insight_strip(
                f"<strong>Total spending trend:</strong> Part B drug spending changed by "
                f"<strong>{growth_pct:+.1f}%</strong> from {int(first_row['Year'])} "
                f"to {int(last_row['Year'])}."
            )
    else:
        only_row = yearly_spending.iloc[0]
        insight_strip(
            f"<strong>Total spending in {int(only_row['Year'])}:</strong> The current filters "
            f"include <strong>{_fmt_cost(only_row['Total Spending'])}</strong> in Part B drug spending."
        )
    chart_card(render_yearly_spending_chart(yearly_spending))

st.divider()

section_heading("Annual top drugs")
drug_ctrl_cols = st.columns([3, 1])
with drug_ctrl_cols[0]:
    top_n = render_top_n_control(
        "Show drugs appearing in each year's top:",
        "ptb_top_n",
    )
with drug_ctrl_cols[1]:
    chart_type = st.segmented_control(
        "Chart type",
        options=["Bar", "Treemap"],
        default="Treemap",
        key="ptb_chart_type",
    )

top_drugs = (
    filtered.groupby(drug_col, as_index=False)["Total Spending"].sum()
    .sort_values("Total Spending", ascending=False)
    .head(top_n)
)
top_names = top_drugs[drug_col].tolist()
n_drugs = filtered[drug_col].nunique()

st.caption(
    f"A drug is included if it ranks in the top {top_n} for any selected year. "
    "The chart then shows that drug's full trend across all selected years."
)

if chart_type == "Bar":
    top_with_year = (
        filtered[filtered[drug_col].isin(top_names)]
        .groupby(["Year", drug_col], as_index=False)["Total Spending"].sum()
    )
    top_with_year["Year"] = top_with_year["Year"].astype(str)
    ordered_years = sorted(top_with_year["Year"].unique())
    color_map = {y: PALETTE_YEARS[i % len(PALETTE_YEARS)] for i, y in enumerate(ordered_years)}
    top_with_year["_text"] = top_with_year["Total Spending"].apply(_fmt_cost)
    fig = px.bar(
        top_with_year,
        x="Total Spending",
        y=drug_col,
        color="Year",
        category_orders={drug_col: top_names, "Year": ordered_years},
        orientation="h",
        template="plotly_white",
        color_discrete_map=color_map,
        barmode="group",
        custom_data=["_text"],
    )
    drug_tickvals, drug_ticktext = _currency_axis_ticks(top_with_year["Total Spending"].max())
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<extra>%{fullData.name}</extra>",
    )
    fig.update_layout(
        height=max(380, 38 * len(top_names)),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(
            title="Total Spending",
            tickmode="array",
            tickvals=drug_tickvals,
            ticktext=drug_ticktext,
        ),
        margin=dict(t=20, l=10, r=10, b=10),
    )
    chart_card(fig)
else:
    treemap_df = top_drugs.copy()
    treemap_df["_text"] = treemap_df["Total Spending"].apply(_fmt_cost)
    fig = px.treemap(
        treemap_df,
        path=[drug_col],
        values="Total Spending",
        color=drug_col,
        color_discrete_sequence=PALETTE_DRUGS,
        custom_data=["_text"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]}",
        hovertemplate="<b>%{label}</b><br>Total Spending: %{customdata[0]}<extra></extra>",
    )
    fig.update_layout(margin=dict(t=16, l=10, r=10, b=10), height=520)
    drug_others = compute_others_stats(filtered, drug_col, top_n)
    if drug_others["count"] > 0:
        drug_layout_cols = st.columns([4, 1], vertical_alignment="top")
        with drug_layout_cols[0]:
            chart_card(fig)
        with drug_layout_cols[1]:
            render_others_card(
                drug_others, top_n, label_singular="drug", accent="#5a2ea8",
            )
    else:
        chart_card(fig)

st.divider()

section_heading("Annual top specialties")
try:
    spec_df_full = load_partb_specialty()
except Exception as exc:  # noqa: BLE001 — file may not yet exist on HF
    st.info(
        "Specialty rollup not yet available on Hugging Face. "
        "Run `python scripts/build_partb_prescriber_dataset.py` to generate it."
    )
    spec_df_full = None

if spec_df_full is not None and not spec_df_full.empty:
    spec_filtered = spec_df_full[spec_df_full["Year"].isin(selected_years)].copy()

if spec_df_full is not None and not spec_df_full.empty and not spec_filtered.empty:
    spec_ctrl_cols = st.columns([3, 1])
    with spec_ctrl_cols[0]:
        spec_top_n = render_top_n_control(
            "Show specialties appearing in each year's top:",
            "ptb_spec_top_n",
        )
    with spec_ctrl_cols[1]:
        spec_chart_type = st.segmented_control(
            "Chart type",
            options=["Bar", "Treemap"],
            default="Treemap",
            key="ptb_spec_chart_type",
        )

    spec_totals = (
        spec_filtered.groupby("Specialty", as_index=False)["Total Spending"].sum()
        .sort_values("Total Spending", ascending=False)
        .head(spec_top_n)
    )
    spec_top_names = spec_totals["Specialty"].tolist()
    n_specs = spec_filtered["Specialty"].nunique()

    st.caption(
        f"A specialty is included if it ranks in the top {spec_top_n} for any selected year. "
        "Source: CMS Physician PUF (drug HCPCS rows only). Specialty here is the "
        "**rendering** clinician's specialty — i.e., who administered the drug and billed "
        "Medicare — not the ordering clinician. So this view shows where Part B drugs "
        "are physically given (Ophthalmology offices, infusion suites, etc.) rather than "
        "who prescribed them. Totals also do not include facility-billed administrations "
        "(hospital outpatient), so they are lower than the Med B Drug Spending totals above."
    )

    if spec_chart_type == "Bar":
        spec_with_year = (
            spec_filtered[spec_filtered["Specialty"].isin(spec_top_names)]
            .groupby(["Year", "Specialty"], as_index=False)["Total Spending"].sum()
        )
        spec_with_year["Year"] = spec_with_year["Year"].astype(str)
        spec_ordered_years = sorted(spec_with_year["Year"].unique())
        spec_color_map = {
            y: PALETTE_YEARS[i % len(PALETTE_YEARS)]
            for i, y in enumerate(spec_ordered_years)
        }
        spec_with_year["_text"] = spec_with_year["Total Spending"].apply(_fmt_cost)
        spec_fig = px.bar(
            spec_with_year,
            x="Total Spending",
            y="Specialty",
            color="Year",
            category_orders={"Specialty": spec_top_names, "Year": spec_ordered_years},
            orientation="h",
            template="plotly_white",
            color_discrete_map=spec_color_map,
            barmode="group",
            custom_data=["_text"],
        )
        spec_tickvals, spec_ticktext = _currency_axis_ticks(spec_with_year["Total Spending"].max())
        spec_fig.update_traces(
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<extra>%{fullData.name}</extra>",
        )
        spec_fig.update_layout(
            height=max(380, 38 * len(spec_top_names)),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                title="Total Spending",
                tickmode="array",
                tickvals=spec_tickvals,
                ticktext=spec_ticktext,
            ),
            margin=dict(t=20, l=10, r=10, b=10),
        )
        chart_card(spec_fig)
    else:
        spec_treemap_df = spec_totals.copy()
        spec_treemap_df["_text"] = spec_treemap_df["Total Spending"].apply(_fmt_cost)
        spec_fig = px.treemap(
            spec_treemap_df,
            path=["Specialty"],
            values="Total Spending",
            color="Specialty",
            color_discrete_sequence=PALETTE_DRUGS,
            custom_data=["_text"],
        )
        spec_fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]}",
            hovertemplate="<b>%{label}</b><br>Total Spending: %{customdata[0]}<extra></extra>",
        )
        spec_fig.update_layout(margin=dict(t=16, l=10, r=10, b=10), height=520)
        spec_others_stats = compute_others_stats(spec_filtered, "Specialty", spec_top_n)
        if spec_others_stats["count"] > 0:
            spec_layout_cols = st.columns([4, 1], vertical_alignment="top")
            with spec_layout_cols[0]:
                chart_card(spec_fig)
            with spec_layout_cols[1]:
                render_others_card(
                    spec_others_stats, spec_top_n,
                    label_singular="specialty",
                    label_plural="specialties",
                    accent="#1d9e75",
                )
        else:
            chart_card(spec_fig)

st.divider()

section_heading("Yearly trend for selected drugs")
trend_options = top_names
trend_picks = st.multiselect(
    "Drugs to plot (defaults to current top list)",
    sorted(set(trend_options)),
    default=trend_options[:5],
    max_selections=8,
)
if trend_picks:
    trend_df = (
        filtered[filtered[drug_col].isin(trend_picks)]
        .groupby(["Year", drug_col], as_index=False)["Total Spending"].sum()
    )
    trend_df["_text"] = trend_df["Total Spending"].apply(_fmt_cost)
    trend_fig = px.line(
        trend_df.sort_values([drug_col, "Year"]),
        x="Year",
        y="Total Spending",
        color=drug_col,
        markers=True,
        template="plotly_white",
        color_discrete_sequence=PALETTE_DRUGS,
        custom_data=["_text"],
    )
    trend_tickvals, trend_ticktext = _currency_axis_ticks(trend_df["Total Spending"].max())
    trend_fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{customdata[0]}<extra></extra>",
    )
    trend_fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(
            title="Total Spending",
            tickmode="array",
            tickvals=trend_tickvals,
            ticktext=trend_ticktext,
        ),
        margin=dict(t=20, l=10, r=10, b=10),
        height=420,
    )
    chart_card(trend_fig)

st.divider()

section_heading("Drill-down: full detail by HCPCS code")
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
render_detail_table(
    display,
    primary_metric="Total Spending",
    entity_col="Brand Name",
    height=520,
)
