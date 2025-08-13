import pytz
from datetime import datetime
import logging
import sqlite3
from twilio_tools import call_customer
import os
import aiosqlite
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd
import json
import re
from typing import List
# from google.adk.agents import Agent
# from google.adk.models.lite_llm import LiteLlm

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dbPath = os.path.join(BASE_DIR, "customers.db")


def time_from_hhmm(s: str) -> time:
    """HH:MM -> time object"""
    return datetime.strptime(s, "%H:%M").time()


def within_window(now_time: time, start: time, end: time) -> bool:
    """
    Return True if now_time is within [start, end], with support for wrap-around.
    e.g. start=22:00 end=02:00 => True if now >=22:00 OR now <= 02:00
    """
    if start <= end:
        return start <= now_time <= end
    # wrap-around midnight
    return now_time >= start or now_time <= end


# Assuming you have helper functions
# time_from_hhmm(hhmm: str) -> datetime.time
# within_window(current: datetime.time, start: datetime.time, end: datetime.time) -> bool

  # adjust import if your LiteLlm path differs

# ---- Fallback tool ----

def _infer_requested_columns(user_query: str, available_cols: List[str]) -> List[str]:
    """
    Heuristically map natural-language terms to DB columns; return the subset present.
    """
    q = user_query.lower()

    aliases = {
        "name": ["name", "customer name", "who"],
        "phone": ["phone", "mobile", "contact", "number"],
        "amount_due": ["amount due", "balance", "balance due", "due", "outstanding", "payment due"],
        "days_past_due": ["days past due", "dpd", "overdue days", "late by"],
        "best_call_day": ["best call day", "best day", "day to call"],
        "best_call_start": ["best call start", "call start"],
        "best_call_end": ["best call end", "call end"],
        "timezone": ["timezone", "tz"],
        "risk_score": ["risk", "risk score", "risk rating"],
        "missed_payment": ["missed", "defaulted", "missed payment"],
        "how_long": ["how long", "tenure", "duration"],
        "state": ["state", "region"],
        "preferred_contact_method": ["preferred contact", "contact method"],
        "call_attempts_count": ["call attempts", "attempts"],
        "successful_contact_rate": ["success rate", "contact rate"],
        "last_contact_date": ["last contact", "last contacted"],
    }

    requested = set()
    for col, terms in aliases.items():
        if any(t in q for t in terms) and col in available_cols:
            requested.add(col)

    # If name-based query (e.g., "show Ram" / "details for Venkat"), keep richer set
    if re.search(r"\bfor\s+\w+|\bof\s+\w+|\bdetails\b|\bprofile\b|\bpersona\b", q):
        for col in ["name", "phone", "timezone", "best_call_start", "best_call_end",
                    "best_call_day", "amount_due", "days_past_due", "risk_score",
                    "how_long", "State", "missed_payment", "preferred_contact_method",
                    "call_attempts_count", "successful_contact_rate", "last_contact_date"]:
            if col in available_cols:
                requested.add(col)

    # Default compact columns if nothing inferred
    if not requested:
        for col in ["name", "phone", "amount_due", "days_past_due", "best_call_day", "risk_score"]:
            if col in available_cols:
                requested.add(col)

    # Preserve original casing if present (e.g., "State")
    out = []
    for c in available_cols:
        lc = c.lower()
        if lc in requested or c in requested:
            out.append(c)
    return out


def default_customers_fallback2(user_query: str,
                                db_path: str = "customers.db",
                                model_id: str = "openai/gpt-4.1",
                                max_rows: int = 200) -> str:
    """
    Fallback tool: loads customers table, converts to pandas, and lets the LLM
    answer the user_query strictly using provided rows.
    Returns Markdown.
    """
    logger.info("Running fallback tool with user query: %s", user_query)
    # 1) Load from SQLite -> DataFrame
    sql = """
    SELECT name, phone, missed_payment, best_call_start, best_call_end, best_call_day,
           timezone, dst_observed, how_long, State, last_contact_date, preferred_contact_method,
           call_attempts_count, successful_contact_rate, amount_due, days_past_due, risk_score
    FROM customers
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        return f" Could not read database: `{e}`"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if df.empty:
        return "No customer records found."

    # 2) Pick useful/requested columns
    requested_cols = _infer_requested_columns(user_query, list(df.columns))
    view_df = df[requested_cols].copy()

    # 3) Limit rows to keep prompt small
    view_df = view_df.head(max_rows)

    # 4) Convert to JSON rows for the model (safer than huge tables)
    records_json = json.dumps(view_df.to_dict(
        orient="records"), ensure_ascii=False)

    # 5) Build instruction for the model
    sys_prompt = (
        "You are a precise data analyst. Answer the user's question ONLY using the provided JSON rows. "
        "If the question asks for a list or table, output a clean Markdown table with appropriate columns. "
        "If you filter or sort, explain briefly what you did. "
        "If the answer is not derivable from the provided rows, say so explicitly. "
        "Do NOT hallucinate values not present in the data. "
        "If there are many rows, summarize clearly (top items, totals, averages) as requested."
    )
    data_note = (
        "Schema (subset): name, phone, missed_payment, best_call_start, best_call_end, best_call_day, "
        "timezone, dst_observed, how_long, State, last_contact_date, preferred_contact_method, "
        "call_attempts_count, successful_contact_rate, amount_due, days_past_due, risk_score."
    )

    user_msg = (
        f"User query:\n{user_query}\n\n"
        f"Data rows (JSON, up to {len(view_df)} rows):\n{records_json}\n\n"
        f"{data_note}"
    )
    import litellm

    # 6) Call the model
    llm = litellm(model=model_id)
    try:
        llm_resp = llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ])
        resp = llm_resp.output_text.strip()
    except Exception as e:
        # Graceful fallback: return a simple preview table
        preview = view_df.head(20)
        # Avoid requiring tabulate; simple CSV block renders fine in Markdown
        csv_preview = preview.to_csv(index=False)
        return (
            f"⚠️ LLM call failed: {e}\n\n"
            f"Showing the first {len(preview)} rows (CSV):\n\n"
            f"```csv\n{csv_preview}\n```"
        )

    return resp


DB_PATH = "customers.db"


def get_all_customers_df():
    """Load all customers from SQLite into a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    return df


def parse_and_filter_customers(user_prompt: str) -> pd.DataFrame:
    """
    Use the LLM to interpret the query and return a filtered DataFrame.
    """
    df = get_all_customers_df()

    logger.info(df)
    # Step 1: Ask LiteLlm model for filters
    # model = LiteLlm(model="openai/gpt-4.1")  # Better reasoning than nano

    df_json = df.to_dict(orient="records")  # Convert to JSON-like list

    # Step 3: Prepare instructions with data context
    system_instruction = f"""
    You are a data query assistant for a loan collections database.
    Here are the available columns:
    name, phone, missed_payment, best_call_start, best_call_end, best_call_day, timezone,
    dst_observed, how_long, State, last_contact_date, preferred_contact_method,
    call_attempts_count, successful_contact_rate, amount_due, days_past_due, risk_score.

    I will provide:
    1. A user query
    2. A sample of the actual customer data.

    You must return **only** a valid JSON object describing the filters to apply.

    Example output:
    {{"Name": "Ram", "Phone": "+19048555482", "State": "New Jersey", "best_call_window": true}}
    """

    import litellm

    resp = litellm.completion(
        model="openai/gpt-4.1",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user",
                "content": f"User Query: {user_prompt}\n\nCustomer Data Sample:\n{json.dumps(df_json, indent=2)}"}
        ],
        max_tokens=200,
        # temperature=0.2,
    )

    # ✅ Extract text content correctly
    llm_response = resp["choices"][0]["message"]["content"].strip()

    try:
        logger.info(f"LLM response for filters: {llm_response}")
        filters = json.loads(llm_response)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", llm_response)
        filters = {}

    logger.info("Inferred filters:", filters)
    # Step 2: Apply State filter
    if "State" in filters and "State" in df.columns:
        target_state = filters["State"].strip().lower()
        logger.info(f"Applying State filter: '{target_state}'")

        # Strip whitespace and lower-case all state names in df
        df["State_clean"] = df["State"].astype(str).str.strip().str.lower()
        logger.info(f"Available States: {df['State_clean'].unique()}")

        # Apply filter on cleaned column
        df = df[df["State_clean"] == target_state]
        logger.info(f"Filtered DataFrame shape: {df.shape}")

        # Drop helper column if needed
    df.drop(columns=["State_clean"], inplace=True)

    # Step 3: Apply name filter
    if "name" in filters and "name" in df.columns:
        df = df[df["name"].str.lower().str.contains(filters["name"].lower())]

    # Step 4: Apply best call window filter
    if (
        filters.get("best_call_window")
        and {"best_call_start", "best_call_end", "timezone"} <= set(df.columns)
    ):
        now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
        matching_rows = []
        for _, row in df.iterrows():
            try:
                tz = pytz.timezone(row["timezone"])
                local_now = now_utc.astimezone(tz)
                start = datetime.strptime(
                    row["best_call_start"], "%H:%M").time()
                end = datetime.strptime(row["best_call_end"], "%H:%M").time()
                if start <= local_now.time() <= end:
                    matching_rows.append(row)
            except Exception:
                continue
        df = pd.DataFrame(matching_rows)

    return df


def default_customers_fallback(user_prompt: str) -> str:
    """
    Fallback method when no other tool matches.
    """
    logger.info("Running fallback tool with user query: %s", user_prompt)
    df = parse_and_filter_customers(user_prompt)
    logger.info(f"Filtered DataFrame shape: {df}")

    if df.empty:
        logger.error("No matching customers found for query: %s", user_prompt)
        return "No matching customers found."

    display_cols = [
        "name", "phone", "amount_due", "days_past_due",
        "best_call_day", "risk_score", "State"
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    logging.info(f"Available columns for display: {available_cols}")
    if not available_cols:
        return "No relevant columns found in the customer data."
    return df[available_cols].to_markdown(index=False)


def get_customer_info(name: str) -> str:
    """
    Fetches detailed customer information from the database by name.
    """
    logger.info(f"Fetching info for customer: {name}")
    conn = sqlite3.connect("customers.db")
    c = conn.cursor()
    c.execute("""
        SELECT name, phone, missed_payment, best_call_start, best_call_end, best_call_day,
               timezone, dst_observed, how_long, State, last_contact_date, preferred_contact_method,
               call_attempts_count, successful_contact_rate, amount_due, days_past_due, risk_score
        FROM customers
        WHERE name LIKE ?
    """, (f"%{name}%",))
    result = c.fetchone()
    conn.close()

    if result:
        return (
            f"Customer: {result[0]}\n"
            f"Phone: {result[1]}\n"
            f"Missed Payment: {result[2]}\n"
            f"Best Call Time: {result[3]} - {result[4]} ({result[5]})\n"
            f"Timezone: {result[6]} | DST Observed: {result[7]}\n"
            f"How Long: {result[8]} mins | State: {result[9]}\n"
            f"Last Contact: {result[10] or 'N/A'}\n"
            f"Preferred Contact: {result[11]}\n"
            f"Call Attempts: {result[12]} | Success Rate: {result[13]:.2f}\n"
            f"Amount Due: ${result[14]:.2f} | Days Past Due: {result[15]}\n"
            f"Risk Score: {result[16]}"
        )
    else:
        return f"No customer found with name '{name}'."


def list_defaulted_customers() -> str:
    """
    Lists all customers who have missed payments with more detail.
    """
    logger.info("Listing all defaulted customers...")
    conn = sqlite3.connect("customers.db")
    c = conn.cursor()
    c.execute("""
        SELECT  name, phone, timezone, best_call_start, best_call_end,
        best_call_day, amount_due, days_past_due, risk_score, how_long, State, missed_payment
        FROM customers
        WHERE missed_payment = 1
        ORDER BY amount_due DESC
    """)
    results = c.fetchall()
    conn.close()

    if not results:
        return "No customers with outstanding dues."

    return "\n".join([
        (
            f"{name} | Phone: {phone} | Timezone: {timezone} | "
            f"Best Call Window: {best_call_start} - {best_call_end} | Best Call Day: {best_call_day} | "
            f"Amount Due: ${amount_due:.2f} | Days Past Due: {days_past_due} | "
            f"Risk Score: {risk_score} | How Long: {how_long} | State: {state} | "
            f"Missed Payment: {'Yes' if missed_payment else 'No'}"
        )
        for name, phone, timezone, best_call_start, best_call_end,
        best_call_day, amount_due, days_past_due, risk_score, how_long, state, missed_payment in results
    ])


def call_customer_by_name(name: str, message: str = "This is a reminder about your pending loan dues.") -> str:
    """
    Calls a customer by name with a message.
    """
    logger.info(f"Calling customer: {name}")
    conn = sqlite3.connect(dbPath)
    c = conn.cursor()
    c.execute("SELECT phone FROM customers WHERE name LIKE ?", (f"%{name}%",))
    result = c.fetchone()
    conn.close()

    if not result:
        return f"No customer found with name '{name}'."

    phone = result[0]
    logging.info(f"Calling {name} at {phone}...")
    return call_customer(name=name, to_number=phone, message=message)


async def get_customers_to_call() -> list:
    """
    Returns a list of  customers who:
      - missed_payment = 1
      - current local time in their timezone is within best_call_start..best_call_end
    """
    print("Fetching customers to call...")
    out = []
    async with aiosqlite.connect(dbPath) as db:
        db.row_factory = aiosqlite.Row
        q = """
        SELECT id, name, phone, missed_payment, best_call_start, best_call_end,
               timezone, best_call_day, amount_due, days_past_due, risk_score
        FROM customers
        WHERE missed_payment = 1
        """
        async with db.execute(q) as cur:
            rows = await cur.fetchall()
            for r in rows:
                tz_str = r["timezone"]
                try:
                    tz = ZoneInfo(tz_str)
                except Exception:
                    continue
                now_local = datetime.now(tz).time()
                start = time_from_hhmm(r["best_call_start"])
                end = time_from_hhmm(r["best_call_end"])
                if within_window(now_local, start, end):
                    out.append({
                        "id": r["id"],
                        "name": r["name"],
                        "phone": r["phone"],
                        "timezone": tz_str,
                        "best_call_start": r["best_call_start"],
                        "best_call_end": r["best_call_end"],
                        "best_call_day": r["best_call_day"],
                        "amount_due": r["amount_due"],
                        "days_past_due": r["days_past_due"],
                        "risk_score": r["risk_score"]
                    })
    return out



async def get_customers_() -> list:
    """
    Returns a list of all customers with their details."""
    print("Fetching customers to call...")
    out = []
    async with aiosqlite.connect(dbPath) as db:
        db.row_factory = aiosqlite.Row
        q = """
        SELECT id, name, phone, missed_payment, best_call_start, best_call_end,
               timezone, best_call_day, amount_due, days_past_due, risk_score
        FROM customers
        """
        async with db.execute(q) as cur:
            rows = await cur.fetchall()
            for r in rows:
                tz_str = r["timezone"]
                out.append({
                    "id": r["id"],
                    "name": r["name"],
                    "phone": r["phone"],
                    "timezone": tz_str,
                    "best_call_start": r["best_call_start"],
                    "best_call_end": r["best_call_end"],
                    "best_call_day": r["best_call_day"],
                    "amount_due": r["amount_due"],
                    "days_past_due": r["days_past_due"],
                    "risk_score": r["risk_score"],
                    "missed_payment": r["missed_payment"]
                })
        logger.info(f"Fetched {len(out)} customers from the database.")        
    return out
