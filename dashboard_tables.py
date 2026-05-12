from __future__ import annotations

from html import escape
from typing import Iterable

import pandas as pd
import streamlit as st


MONEY_COLUMNS = {
    "Total Drug Cost",
    "Total Spending",
    "Cost per Claim",
    "Cost per 30-Day Fill",
    "Avg Spending per Dosage Unit",
    "Avg Spending per Beneficiary",
    "Avg Medicare Payment",
}
COUNT_COLUMNS = {
    "Total Claims",
    "Total 30-Day Fills",
    "Total Days Supply",
    "Total Dosage Units",
    "Total Beneficiaries",
    "Total Services",
}
LONG_TEXT_COLUMNS = {
    "HCPCS Description",
    "Drug Name",
    "Therapeutic Class",
    "Specialty",
    "Prescriber Name",
    "Rendering Provider",
}
ID_TEXT_COLUMNS = {"State", "HCPCS Code", "Prescriber NPI", "Rendering NPI"}
ENTITY_COLUMNS = [
    "Drug Name",
    "Therapeutic Class",
    "Specialty",
    "Brand Name",
    "Generic Name",
    "HCPCS Description",
    "Prescriber Name",
    "Rendering Provider",
]


def install_table_styles() -> None:
    st.markdown(
        """
<style>
[data-testid="stDataFrame"] {
    border: 1px solid #e6ebf2;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
[data-testid="stDataFrame"] table {
    font-size: 12px !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}
[data-testid="stDataFrame"] th {
    background: #f8fafc !important;
    color: #475569 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: .03em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
[data-testid="stDataFrame"] td {
    border-bottom: 1px solid #f1f5f9 !important;
    color: #111827 !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: #f8fbff !important;
}
.modern-table-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin: 10px 0 8px;
    color: #475569;
    font-size: 12px;
}
.modern-table-summary span {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 4px 9px;
    line-height: 1.3;
}
div[data-testid="stExpander"] {
    border-color: #e2e8f0 !important;
    border-radius: 8px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_chart_detail_table(
    df: pd.DataFrame,
    *,
    label: str = "View detailed rows",
    primary_metric: str | None = None,
    entity_col: str | None = None,
    height: int | None = None,
) -> None:
    """Render a low-noise table behind an expander below a chart."""
    install_table_styles()
    _render_summary(df, primary_metric=primary_metric, entity_col=entity_col)
    with st.expander(label, expanded=False):
        render_table(
            df,
            primary_metric=primary_metric,
            entity_col=entity_col,
            height=height,
            show_share=primary_metric is not None,
        )


def render_detail_table(
    df: pd.DataFrame,
    *,
    primary_metric: str | None = None,
    entity_col: str | None = None,
    height: int | None = None,
) -> None:
    """Render an always-visible drill-down table."""
    install_table_styles()
    _render_summary(df, primary_metric=primary_metric, entity_col=entity_col)
    render_table(
        df,
        primary_metric=primary_metric,
        entity_col=entity_col,
        height=height,
        show_share=primary_metric is not None,
    )


def render_results_table(
    df: pd.DataFrame,
    *,
    primary_metric: str | None = None,
    entity_col: str | None = None,
    height: int | None = None,
) -> None:
    """Render an always-visible ranked result table for provider search."""
    install_table_styles()
    render_table(
        df,
        primary_metric=primary_metric,
        entity_col=entity_col,
        height=height,
        show_share=primary_metric is not None,
        add_rank=True,
    )


def render_table(
    df: pd.DataFrame,
    *,
    primary_metric: str | None = None,
    entity_col: str | None = None,
    height: int | None = None,
    show_share: bool = False,
    add_rank: bool = False,
) -> None:
    if df.empty:
        st.info("No rows to display.")
        return

    table_df = _prepare_dataframe(
        df,
        primary_metric=primary_metric,
        show_share=show_share,
        add_rank=add_rank,
    )
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=height or _height_for_rows(len(table_df)),
        column_config=_column_config(table_df, primary_metric=primary_metric),
        column_order=list(table_df.columns),
    )


def _prepare_dataframe(
    df: pd.DataFrame,
    *,
    primary_metric: str | None,
    show_share: bool,
    add_rank: bool,
) -> pd.DataFrame:
    table_df = df.copy()
    for col in ID_TEXT_COLUMNS.intersection(table_df.columns):
        table_df[col] = table_df[col].astype("string")
    if add_rank and "Rank" not in table_df.columns:
        table_df.insert(0, "Rank", range(1, len(table_df) + 1))
    if (
        show_share
        and primary_metric
        and primary_metric in table_df.columns
        and "Spend Share" not in table_df.columns
    ):
        total = table_df[primary_metric].sum()
        if total:
            insert_at = min(table_df.columns.get_loc(primary_metric) + 1, len(table_df.columns))
            table_df.insert(insert_at, "Spend Share", table_df[primary_metric] / total * 100)
    return table_df


def _column_config(
    df: pd.DataFrame,
    *,
    primary_metric: str | None,
) -> dict:
    config = {}
    for col in df.columns:
        if col == "Rank":
            config[col] = st.column_config.NumberColumn(col, format="%d", width="small")
        elif col == "Year":
            config[col] = st.column_config.NumberColumn(col, format="%d", width="small")
        elif col in MONEY_COLUMNS:
            help_text = "Primary comparison metric" if col == primary_metric else None
            config[col] = st.column_config.NumberColumn(
                col,
                format="dollar",
                help=help_text,
                width="medium",
            )
        elif col in COUNT_COLUMNS:
            config[col] = st.column_config.NumberColumn(
                col,
                format="localized",
                width="medium",
            )
        elif col == "Spend Share":
            max_value = max(100.0, float(df[col].max()) if not df.empty else 100.0)
            config[col] = st.column_config.ProgressColumn(
                col,
                format="%.1f%%",
                min_value=0,
                max_value=max_value,
                help="Share of the visible table total.",
                width="medium",
            )
        elif col == "Distance (mi)":
            config[col] = st.column_config.NumberColumn(col, format="%.1f", width="small")
        elif col in LONG_TEXT_COLUMNS:
            config[col] = st.column_config.TextColumn(col, width="large")
        elif col in ID_TEXT_COLUMNS:
            config[col] = st.column_config.TextColumn(col, width="small")
    return config


def _render_summary(
    df: pd.DataFrame,
    *,
    primary_metric: str | None,
    entity_col: str | None,
) -> None:
    if df.empty:
        return

    chips = [f"{len(df):,} rows"]
    metric_col = primary_metric if primary_metric in df.columns else None
    if metric_col:
        total = df[metric_col].sum()
        chips.append(f"{metric_col}: {_format_metric(total, metric_col)}")
        entity = entity_col or _first_existing(df.columns, ENTITY_COLUMNS)
        if entity and entity in df.columns:
            top_row = df.sort_values(metric_col, ascending=False).iloc[0]
            top_label = escape(str(top_row[entity]))
            chips.append(f"Top: {top_label} ({_format_metric(top_row[metric_col], metric_col)})")

    html = "".join(f"<span>{chip}</span>" for chip in chips)
    st.markdown(f'<div class="modern-table-summary">{html}</div>', unsafe_allow_html=True)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    existing = set(columns)
    return next((candidate for candidate in candidates if candidate in existing), None)


def _height_for_rows(row_count: int) -> int:
    return min(520, max(220, 38 * min(row_count, 12) + 42))


def _format_metric(value: float, col: str) -> str:
    if col in MONEY_COLUMNS:
        return _format_money(value)
    if col in COUNT_COLUMNS:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def _format_money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:,.0f}"
