from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic
import duckdb
import streamlit as st
from huggingface_hub import hf_hub_download


REPO_DIR = Path(__file__).resolve().parent
HF_DATASET_ID = "mbateya/medicare_part_d_prescribers"

LOCAL_PARTD_PATH = REPO_DIR / "data" / "processed" / "medicare_partd_2021_2023.parquet"
HF_PARTD_FILE = "processed/medicare_partd_2021_2023.parquet"

AI_DATA_FILES = {
    "partb_drugs": "part_b_drug_spending.parquet",
    "partb_specialty": "partb_drug_by_specialty.parquet",
    "partb_state": "partb_drug_by_state.parquet",
    "partb_state_specialty": "partb_drug_by_state_specialty.parquet",
    "drug_atc": "drug_atc.parquet",
    "state_population": "state_population.parquet",
}

STATE_CODES = [
    ("AL", "Alabama"), ("AK", "Alaska"), ("AZ", "Arizona"), ("AR", "Arkansas"),
    ("CA", "California"), ("CO", "Colorado"), ("CT", "Connecticut"), ("DE", "Delaware"),
    ("DC", "District of Columbia"), ("FL", "Florida"), ("GA", "Georgia"), ("HI", "Hawaii"),
    ("ID", "Idaho"), ("IL", "Illinois"), ("IN", "Indiana"), ("IA", "Iowa"),
    ("KS", "Kansas"), ("KY", "Kentucky"), ("LA", "Louisiana"), ("ME", "Maine"),
    ("MD", "Maryland"), ("MA", "Massachusetts"), ("MI", "Michigan"), ("MN", "Minnesota"),
    ("MS", "Mississippi"), ("MO", "Missouri"), ("MT", "Montana"), ("NE", "Nebraska"),
    ("NV", "Nevada"), ("NH", "New Hampshire"), ("NJ", "New Jersey"), ("NM", "New Mexico"),
    ("NY", "New York"), ("NC", "North Carolina"), ("ND", "North Dakota"), ("OH", "Ohio"),
    ("OK", "Oklahoma"), ("OR", "Oregon"), ("PA", "Pennsylvania"), ("RI", "Rhode Island"),
    ("SC", "South Carolina"), ("SD", "South Dakota"), ("TN", "Tennessee"), ("TX", "Texas"),
    ("UT", "Utah"), ("VT", "Vermont"), ("VA", "Virginia"), ("WA", "Washington"),
    ("WV", "West Virginia"), ("WI", "Wisconsin"), ("WY", "Wyoming"),
    ("PR", "Puerto Rico"), ("VI", "U.S. Virgin Islands"), ("GU", "Guam"),
    ("AS", "American Samoa"), ("MP", "Northern Mariana Islands"),
]

CHATBOT_MODEL = "claude-haiku-4-5"
CHATBOT_MAX_TOOL_CALLS = 6
CHATBOT_ROW_LIMIT = 50
CHATBOT_MAX_TOKENS = 4096


CHATBOT_SYSTEM_PROMPT = """\
You are a data analyst for the Medicare drug dashboards. Answer the user's question \
by querying DuckDB views via the run_sql tool.

Available views:

1) partd_drugs — Medicare Part D pharmacy-dispensed prescribing rollup, years 2021-2023.
Columns:
  Year, State, Specialty, "Brand Name", "Generic Name", "Total Drug Cost",
  "Total Claims", "Total 30-Day Fills", "Cost per Claim", "Cost per 30-Day Fill"

2) partb_drugs — official annual Medicare Part B drug spending by HCPCS, years 2019-2023.
Columns:
  Year, "HCPCS Code", "HCPCS Description", "Brand Name", "Generic Name",
  "Total Spending", "Total Dosage Units", "Total Claims", "Total Beneficiaries",
  "Avg Spending per Dosage Unit", "Avg Spending per Claim", "Avg Spending per Beneficiary",
  "Outlier Flag"

3) partb_specialty — Physician PUF Part B drug-HCPCS rollup by rendering specialty, years 2021-2023.
Columns:
  Year, Specialty, "HCPCS Code", "HCPCS Description", "Total Spending",
  "Total Services", "Total Beneficiaries"

4) partb_state — Physician PUF Part B drug-HCPCS rollup by rendering state, years 2021-2023.
Columns:
  Year, State, "HCPCS Code", "HCPCS Description", "Brand Name", "Generic Name",
  "Total Spending", "Total Services", "Total Beneficiaries"

5) partb_state_specialty — Physician PUF Part B drug-HCPCS rollup by rendering state and specialty, years 2021-2023.
Columns:
  Year, State, Specialty, "HCPCS Code", "HCPCS Description", "Total Spending",
  "Total Services", "Total Beneficiaries"

6) drug_atc — Generic Name to ATC classification lookup.
Columns:
  "Generic Name", atc_level_1_code, atc_level_1_name, atc_level_2_code, atc_level_2_name,
  atc_level_3_code, atc_level_3_name, atc_level_4_code, atc_level_4_name

7) state_population — state population lookup for per-capita questions.
Columns:
  State, Population_2021, Population_2022, Population_2023

8) state_codes — state abbreviation/name lookup.
Columns:
  State, "State Name"

9) drug_spending_all — curated cross-dashboard spend view for common comparisons.
Columns:
  Source, Year, State, Specialty, "HCPCS Code", "HCPCS Description", "Brand Name",
  "Generic Name", Spend, Volume, "Volume Metric"
Source values:
  'Part D' = pharmacy-dispensed Part D rollup from partd_drugs
  'Part B Official' = official national Part B drug spending from partb_drugs
  'Part B Physician PUF State' = physician-administered Part B rollup from partb_state
  'Part B Physician PUF Specialty' = physician-administered Part B rollup from partb_specialty
  'Part B Physician PUF State Specialty' = physician-administered Part B rollup from partb_state_specialty

Guidelines:
- Always wrap multi-word column names in double quotes.
- LIMIT final row-returning SELECTs to 50 rows.
- After receiving query results, write a 1-3 sentence answer in plain English.
- Use part-specific metric names when answering: Part D uses "Total Drug Cost"; official Part B uses "Total Spending"; Physician PUF rollups use "Total Spending" and "Total Services".
- For cross-Part comparisons, prefer drug_spending_all unless a source-specific table is clearly better.
- Official Part B spending is national only. Use partb_state or partb_state_specialty for state-level Part B questions, and explain that those are Physician PUF rendering-provider rollups that exclude facility-billed administrations.
- For Part B state rollups, join State to state_codes.State when full state names or population joins are needed.
- Provider Search raw partitions are not available to Ask AI. If asked for individual providers near a city, by radius, or by provider name, say that this requires the Provider Search page and do not attempt SQL over provider partitions.
- Do not invent column names or values that are not in the schemas above.

Drug name matching:
- Match drug names with ILIKE '%term%' across "Brand Name", "Generic Name", and for Part B when relevant "HCPCS Description".
- Avoid exact equality for drug names unless the user explicitly asks for an exact string.
- Aggregate across matching rows so formulation/brand variants combine.
- If no rows or NULL are returned, retry once with a shorter substring before saying data is unavailable.

Specialty and state matching:
- Use ILIKE '%term%' on Specialty for specialty names.
- Part D State uses full names such as 'California'. Part B state rollups may use abbreviations; inspect distinct values if uncertain.
"""


RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": (
        "Execute a read-only SELECT or WITH query against dashboard DuckDB views "
        "and return the rows."
    ),
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

_FORBIDDEN_SQL_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "ATTACH", "DETACH", "COPY", "PRAGMA", "EXPORT", "IMPORT",
)


@st.cache_resource(show_spinner=False)
def _get_chatbot_client() -> anthropic.Anthropic | None:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource(show_spinner=False)
def _resolved_partd_path() -> str:
    if LOCAL_PARTD_PATH.exists():
        return LOCAL_PARTD_PATH.as_posix()
    return hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_PARTD_FILE,
        repo_type="dataset",
    )


@st.cache_resource(show_spinner=False)
def _resolved_hf_path(filename: str) -> str:
    return hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=filename,
        repo_type="dataset",
    )


@st.cache_resource(show_spinner="Preparing Ask AI data...")
def _get_ai_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW partd_drugs AS SELECT * FROM read_parquet('{_resolved_partd_path()}')")
    for view_name, filename in AI_DATA_FILES.items():
        path = _resolved_hf_path(filename)
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")
    state_values = ", ".join(f"('{code}', '{name}')" for code, name in STATE_CODES)
    con.execute(
        f"""
        CREATE VIEW state_codes AS
        SELECT * FROM (VALUES {state_values}) AS t(State, "State Name")
        """
    )
    con.execute(
        """
        CREATE VIEW partb_hcpcs_lookup AS
        SELECT
            Year,
            "HCPCS Code",
            STRING_AGG(DISTINCT "Brand Name", '; ' ORDER BY "Brand Name") AS "Brand Name",
            STRING_AGG(DISTINCT "Generic Name", '; ' ORDER BY "Generic Name") AS "Generic Name"
        FROM partb_drugs
        WHERE "HCPCS Code" IS NOT NULL
        GROUP BY Year, "HCPCS Code"
        """
    )
    con.execute(
        """
        CREATE VIEW drug_spending_all AS
        SELECT
            'Part D' AS Source,
            Year,
            State,
            Specialty,
            CAST(NULL AS VARCHAR) AS "HCPCS Code",
            CAST(NULL AS VARCHAR) AS "HCPCS Description",
            "Brand Name",
            "Generic Name",
            "Total Drug Cost" AS Spend,
            "Total Claims" AS Volume,
            'Total Claims' AS "Volume Metric"
        FROM partd_drugs
        UNION ALL
        SELECT
            'Part B Official' AS Source,
            Year,
            CAST(NULL AS VARCHAR) AS State,
            CAST(NULL AS VARCHAR) AS Specialty,
            "HCPCS Code",
            "HCPCS Description",
            "Brand Name",
            "Generic Name",
            "Total Spending" AS Spend,
            "Total Claims" AS Volume,
            'Total Claims' AS "Volume Metric"
        FROM partb_drugs
        UNION ALL
        SELECT
            'Part B Physician PUF State' AS Source,
            Year,
            State,
            CAST(NULL AS VARCHAR) AS Specialty,
            "HCPCS Code",
            "HCPCS Description",
            "Brand Name",
            "Generic Name",
            "Total Spending" AS Spend,
            "Total Services" AS Volume,
            'Total Services' AS "Volume Metric"
        FROM partb_state
        UNION ALL
        SELECT
            'Part B Physician PUF Specialty' AS Source,
            ps.Year,
            CAST(NULL AS VARCHAR) AS State,
            ps.Specialty,
            ps."HCPCS Code",
            ps."HCPCS Description",
            lookup."Brand Name",
            lookup."Generic Name",
            ps."Total Spending" AS Spend,
            ps."Total Services" AS Volume,
            'Total Services' AS "Volume Metric"
        FROM partb_specialty ps
        LEFT JOIN partb_hcpcs_lookup lookup
          ON ps.Year = lookup.Year
         AND ps."HCPCS Code" = lookup."HCPCS Code"
        UNION ALL
        SELECT
            'Part B Physician PUF State Specialty' AS Source,
            pss.Year,
            pss.State,
            pss.Specialty,
            pss."HCPCS Code",
            pss."HCPCS Description",
            lookup."Brand Name",
            lookup."Generic Name",
            pss."Total Spending" AS Spend,
            pss."Total Services" AS Volume,
            'Total Services' AS "Volume Metric"
        FROM partb_state_specialty pss
        LEFT JOIN partb_hcpcs_lookup lookup
          ON pss.Year = lookup.Year
         AND pss."HCPCS Code" = lookup."HCPCS Code"
        """
    )
    return con


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
        result = _get_ai_duckdb().execute(sql_clean).fetchdf()
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


def _escape_dollars(text: str) -> str:
    return re.sub(r"(?<!\\)\$", r"\\$", text)


@st.dialog("Ask AI", width="large")
def open_chatbot_dialog() -> None:
    render_chatbot()


def render_chatbot_button() -> None:
    if st.button(
        "Ask AI",
        icon=":material/auto_awesome:",
        use_container_width=True,
    ):
        open_chatbot_dialog()


def render_chatbot() -> None:
    st.markdown(
        "Ask a question across the Med D, Med B, and Physician PUF dashboard rollups. "
        "Provider-level radius/name searches still live on the Provider Search page."
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
        "Compare Part D and Part B spending for Keytruda by year.",
        "Which state had the highest Part B physician-administered spending in 2023?",
        "What therapeutic classes drove Part D spending in 2023?",
    ]
    pending_input: str | None = None
    for col, suggestion in zip(suggestion_cols, suggestions):
        with col:
            if st.button(suggestion, key=f"chat_suggest_{suggestion}", use_container_width=True):
                pending_input = suggestion

    user_input = st.chat_input("Ask about Medicare drug dashboard rollups")
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
                        f"_Underlying error - **{err_type}**: `{err_str}`_"
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
