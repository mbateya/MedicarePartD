from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st


APP_TITLE = "Medicare Part D Prescribing Dashboard"
APP_SUBTITLE = (
    "Interactive analysis of Medicare Part D prescribing patterns by year, drug, "
    "specialty, and state."
)

REPO_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = REPO_DIR / "data" / "processed" / "medicare_partd_2021_2023.parquet"
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
}


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")


METRIC_CARD_CSS = """
<style>
div.metric-card {
    background: linear-gradient(180deg, #f4fbf9 0%, #eef7f4 100%);
    border: 1px solid #d9ebe4;
    border-radius: 16px;
    padding: 1rem 1.1rem 0.95rem 1.1rem;
    min-height: 120px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div.metric-card.metric-card-alt {
    background: linear-gradient(180deg, #f7fbfd 0%, #eef5f8 100%);
    border-color: #dbe7ee;
}
div.metric-label {
    color: #5f6b76;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
div.metric-value {
    color: #143a31;
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1.05;
}
div.metric-card-alt div.metric-value {
    color: #1f3442;
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


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Specialty"] = _normalize_specialties(df["Specialty"])
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
    if PROCESSED_PATH.exists():
        return _normalize_dataset(
            _load_processed_parquet(
                PROCESSED_PATH,
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


def summarize_trends(df: pd.DataFrame, grouping: str, selected_drugs: list[str]) -> pd.DataFrame:
    drug_col = "Brand Name" if grouping == "Brand name" else "Generic Name"
    trend_df = df[df[drug_col].isin(selected_drugs)]
    summary = _summarize(trend_df, ["Year", drug_col])
    return summary.rename(columns={drug_col: "Drug Name"}).sort_values(["Drug Name", "Year"])


def format_tables(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    currency_cols = [
        col
        for col in ["Total Drug Cost", "Cost per Claim", "Cost per 30-Day Fill"]
        if col in df.columns
    ]
    number_cols = [
        col
        for col in ["Total Claims", "Total 30-Day Fills", "Total Days Supply"]
        if col in df.columns
    ]
    formats = {col: "${:,.2f}" for col in currency_cols}
    formats.update({col: "{:,.0f}" for col in number_cols})
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


def _format_filter_chip(label: str, values: list[str] | list[int]) -> str:
    preview = ", ".join(map(str, values[:4]))
    suffix = "..." if len(values) > 4 else ""
    return f'<span class="filter-chip">{label}: {preview}{suffix}</span>'


def render_metric_cards(total_cost: float, total_claims: float) -> None:
    cards = [
        ("Total Drug Cost", f"${total_cost:,.0f}", ""),
        ("Total Claims", f"{total_claims:,.0f}", "metric-card-alt"),
    ]
    columns = st.columns(2)
    for col, (label, value, variant) in zip(columns, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card {variant}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_filter_summary(years: list[int], states: list[str], specialties: list[str]) -> None:
    chips = []
    if years:
        chips.append(_format_filter_chip("Year", years))
    if states:
        chips.append(_format_filter_chip("State", states))
    if specialties:
        chips.append(_format_filter_chip("Specialty", specialties))

    if not chips:
        chips.append('<span class="filter-chip">Global filters: All records</span>')

    st.markdown(
        f"""
        <div class="filter-summary">
            <div class="filter-summary-title">Applied Global Filters</div>
            <div class="filter-chip-row">
                {''.join(chips)}
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


def render_charts(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    orientation: str = "v",
):
    chart_df = df.copy()
    if color == "Year":
        chart_df[color] = chart_df[color].astype(str)

    fig = px.bar(
        chart_df,
        x=x,
        y=y,
        color=color,
        barmode="group",
        orientation=orientation,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data={
            "Total Drug Cost": ":$,.2f",
            "Total Claims": ":,.0f",
            "Total 30-Day Fills": ":,.0f",
            "Cost per Claim": ":$,.2f",
            "Cost per 30-Day Fill": ":$,.2f",
        },
        title=title,
    )
    fig.update_layout(
        legend_title_text=color,
        title_x=0,
        margin=dict(l=20, r=20, t=70, b=20),
        hoverlabel=dict(bgcolor="white"),
    )
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


def _select_options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def main() -> None:
    st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

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

    year_options = sorted(df["Year"].dropna().astype(int).unique().tolist())
    state_options = _select_options(df["State"])
    specialty_options = _select_options(df["Specialty"])

    st.markdown("#### Filters")
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

    total_cost = filtered_df["Total Drug Cost"].sum()
    total_claims = filtered_df["Total Claims"].sum()

    render_metric_cards(total_cost, total_claims)
    render_filter_summary(selected_years, selected_states, selected_specialties)

    st.divider()

    st.subheader("Annual Top Drugs")
    top_drug_n = render_top_n_control(
        "Show drugs appearing in each year's top:",
        "top_drug_n",
    )
    top_drugs = summarize_top_drugs(filtered_df, grouping, top_drug_n)
    drug_title = _section_title(
        f"Drugs Appearing in the Annual Top {top_drug_n} by Total Drug Cost",
        grouping=grouping,
        context=context,
    )
    st.caption(context)
    st.caption(
        f"A drug is included if it ranks in the top {top_drug_n} for any selected year. "
        "The chart then shows that drug's full trend across all selected years."
    )
    st.plotly_chart(
        render_charts(
            top_drugs,
            x="Drug Name",
            y="Total Drug Cost",
            color="Year",
            title=drug_title,
        ),
        use_container_width=True,
    )
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
    )

    st.divider()

    st.subheader("Annual Top Specialties")
    top_specialty_n = render_top_n_control(
        "Show specialties appearing in each year's top:",
        "top_specialty_n",
    )
    top_specialties = summarize_top_specialties(specialty_section_df, top_specialty_n)
    specialty_context = _filter_context(selected_years, selected_states, [])
    st.caption(specialty_context)
    st.caption(
        f"A specialty is included if it ranks in the top {top_specialty_n} for any "
        "selected year. The chart then shows that specialty's full trend across all "
        "selected years."
    )
    st.plotly_chart(
        render_charts(
            top_specialties,
            x="Specialty",
            y="Total Drug Cost",
            color="Year",
            title=_section_title(
                (
                    "Specialties Appearing in the Annual Top "
                    f"{top_specialty_n} by Total Drug Cost"
                ),
                context=specialty_context,
            ),
        ),
        use_container_width=True,
    )
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
    )

    st.divider()

    st.subheader("Yearly Drug Trend")
    st.write("Select up to 5 drugs and click Generate yearly trend.")

    drug_col = "Brand Name" if grouping == "Brand name" else "Generic Name"
    trend_options = _select_options(filtered_df[drug_col])
    selected_trend_drugs = st.multiselect(
        f"Select {grouping.lower()} drugs",
        trend_options,
        max_selections=5,
    )

    if st.button("Generate yearly trend"):
        if not selected_trend_drugs:
            st.info("Select at least one drug to generate the yearly trend.")
        elif len(selected_trend_drugs) > 5:
            st.warning("Select no more than 5 drugs.")
        else:
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
                title=_section_title(
                    "Yearly trend for selected drugs",
                    grouping=grouping,
                    context=context,
                ),
            )
            fig.update_layout(
                title_x=0,
                legend_title_text="Drug Name",
                margin=dict(l=20, r=20, t=70, b=20),
            )
            tick_vals, tick_text = _build_billions_ticks(float(trend_df["Total Drug Cost"].max()))
            fig.update_yaxes(tickvals=tick_vals, ticktext=tick_text, rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)
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
            )

    st.divider()
    st.caption("Run with: streamlit run Med_D_dashboard.py")
    st.caption(
        "This app is dynamic and must be deployed using Streamlit Cloud or similar. "
        "GitHub Pages cannot run Python apps."
    )


if __name__ == "__main__":
    main()
