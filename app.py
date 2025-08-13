import asyncio
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
from collection_tools import get_customers_to_call, get_customers_
from twilio_tools import call_customer
from run_agent_chat import init_agent, ask_agent
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from io import StringIO
import base64

import threading
import uvicorn

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collections_caller")

app = FastAPI(title="Collections Caller with Agent")


class CallResult(BaseModel):
    to: str
    sid: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class RunCallsResponse(BaseModel):
    called: int
    results: List[CallResult]


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up and initializing agent...")
    await init_agent()
    logger.info("Agent initialized.")


@app.get("/report")
async def report():
    customers = await get_customers_()
    names = [c["name"] for c in customers]
    logger.info(f"Report requested - {len(names)} customers to call.")
    return {"count": len(names), "to_call": names}


@app.post("/run_calls", response_model=RunCallsResponse)
async def run_calls():
    logger.info("Starting to run calls...")
    customers = await get_customers_to_call()
    if not customers:
        logger.info("No customers to call.")
        return RunCallsResponse(called=0, results=[])

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, call_customer, c) for c in customers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for res, c in zip(results, customers):
        if isinstance(res, Exception):
            logger.error(f"Error calling {c['phone']}: {res}")
            out.append(CallResult(to=c["phone"], error=str(res)))
        else:
            out.append(
                CallResult(
                    to=res.get("to"), sid=res.get("sid"), status=res.get("status")
                )
            )

    logger.info(f"Finished running calls: {len(out)} calls processed.")
    return RunCallsResponse(called=len(out), results=out)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Chat query received: {request.query}")
    answer = await ask_agent(request.query)
    logger.info("Chat response sent.")
    return ChatResponse(response=answer)


# === Gradio UI === #

async def gradio_chat_fn1(message, history):
    logger.info(f"Gradio interactive chat: {message}")
    answer = await ask_agent(message)
    history.append((message, answer))
    return answer, history  # return assistant's reply + updated history

async def gradio_chat_fn(message, history):
    logger.info(f"Gradio interactive chat: {message}")
    answer = await ask_agent(message)
    return answer

def gradio_report2(show_graph=False):
    # Fetch customers data
    customers = asyncio.run(get_customers_to_call())

    if not customers:
        return "No customers found", None, None

    # Convert to DataFrame for easy display
    df = pd.DataFrame(customers)

    # Count
    count = len(df)
    logger.info(f"Gradio Report requested - {count} customers")

    # CSV download
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Graph (optional)
    graph = None
    if show_graph:
        # Example: Bar chart of customers by State
        graph = px.bar(df, x="State", title="Customers by State", color="missed_payment")

    # Return
    return f"# Number of customers: {count}", df, csv_data, graph


def gradio_report1(show_graph=False):
    # Fetch customers data
    customers = asyncio.run(get_customers_to_call())

    if not customers:
        return "No customers found", None, None, None

    # Convert to DataFrame
    df = pd.DataFrame(customers)

    # Count
    count = len(df)
    logger.info(f"Gradio Report requested - {count} customers")

    # Prepare CSV download link
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    b64 = base64.b64encode(csv_data.encode()).decode()
    download_link = f'<a href="data:file/csv;base64,{b64}" download="customers_report.csv">📥 Download CSV</a>'

    # Graph (optional)
    graph = None
    if show_graph:
        graph = px.bar(
            df,
            x="State",
            title="Customers by State",
            color="missed_payment",
            barmode="group"
        )

    # Return report text, table, download link, and graph
    return f"# Number of customers: {count}", df, download_link, graph

# Updated report function
def gradio_report(show_graph: bool = False, group_by: str = "best_call_day", chart_type: str = "bar"):
    # Fetch customers (synchronous wrapper)
    customers = asyncio.run(get_customers_())

    if not customers:
        return "# Number of customers: 0", pd.DataFrame(), "<i>No data</i>", None

    # Convert to DataFrame and normalize column names (keep original casing used by your DB)
    df = pd.DataFrame(customers)

    # Ensure numeric columns are numeric (coerce errors -> NaN)
    for num_col in ("amount_due", "days_past_due", "risk_score", "missed_payment"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    count = len(df)
    logger.info(f"Gradio Report requested - {count} customers")

    # CSV download (base64 Data URI)
    buf = StringIO()
    df.to_csv(buf, index=False)
    csv_data = buf.getvalue()
    b64 = base64.b64encode(csv_data.encode()).decode()
    download_link = f'<a href="data:file/csv;base64,{b64}" download="customers_report.csv">📥 Download CSV</a>'

    # Decide valid grouping options (only columns that exist)
    possible_group_cols = [
        c for c in ["best_call_day", "timezone", "missed_payment", "risk_score", "days_past_due", "amount_due", "name"]
        if c in df.columns
    ]

    # Ensure requested group_by is valid; fallback to first available
    if show_graph and group_by not in possible_group_cols:
        group_by = possible_group_cols[0] if possible_group_cols else None

    fig = None
    if show_graph and group_by:
        try:
            # BAR: counts per group
            if chart_type == "bar":
                grouped = df.groupby(group_by).size().reset_index(name="count")
                fig = px.bar(grouped, x=group_by, y="count", title=f"Count by {group_by}")

            # PIE: proportion by group
            elif chart_type == "pie":
                grouped = df.groupby(group_by).size().reset_index(name="count")
                fig = px.pie(grouped, names=group_by, values="count", title=f"Share by {group_by}")

            # HISTOGRAM: numeric distribution (only for numeric columns)
            elif chart_type == "histogram":
                if pd.api.types.is_numeric_dtype(df[group_by]):
                    fig = px.histogram(df, x=group_by, nbins=30, title=f"Distribution of {group_by}")
                else:
                    # fall back to bar counts
                    grouped = df.groupby(group_by).size().reset_index(name="count")
                    fig = px.bar(grouped, x=group_by, y="count", title=f"Count by {group_by}")

            # BOX: show numeric spread of amount_due (or days_past_due) by group
            elif chart_type == "box":
                y_col = "amount_due" if "amount_due" in df.columns else ("days_past_due" if "days_past_due" in df.columns else None)
                if y_col:
                    fig = px.box(df, x=group_by, y=y_col, title=f"{y_col} by {group_by}")
                else:
                    grouped = df.groupby(group_by).size().reset_index(name="count")
                    fig = px.bar(grouped, x=group_by, y="count", title=f"Count by {group_by}")

        except Exception as e:
            logger.exception("Error generating graph, falling back to no graph.")
            fig = None

    # Markdown heading + small summary
    report_md = f"# Number of customers: {count}\n\n"
    #report_md += f"Columns: {', '.join(df.columns.tolist())}\n\n"
    report_md += f"Showing first 10 rows below."

    # Return Markdown text, DataFrame (for table), HTML download link, and Plotly figure (or None)
    return report_md, df, download_link, fig


def update_report(show_graph, group_by, chart_type):
    report_text, df, csv_html, graph = gradio_report(
        show_graph=show_graph,
        group_by=group_by,
        chart_type=chart_type
    )
    
    # Control plot visibility based on checkbox
    if show_graph:
        return report_text, df, csv_html, gr.update(value=graph, visible=True)
    else:
        return report_text, df, csv_html, gr.update(visible=False)


with gr.Blocks(title="Collections Caller", analytics_enabled =False,   css="footer {display: none !important}" ) as demo:
    with gr.Tabs():
        with gr.TabItem("Chat"):
            gr.ChatInterface(
                fn=gradio_chat_fn,
                title="AI Collections Agent",
                description="Chat live with the AI agent."
            )
        with gr.TabItem("Report"):
            # controls
            show_graph_toggle = gr.Checkbox(label="Show Graph", value=False)
            # We'll compute possible group_by options client-side from a fixed list; gradio will ignore invalid selections on the server
            group_by_dropdown = gr.Dropdown(
                choices=["best_call_day", "timezone", "missed_payment", "risk_score", "days_past_due", "amount_due"],
                value="best_call_day",
                label="Group / Aggregate by"
            )
            chart_type_dropdown = gr.Dropdown(
                choices=["bar", "pie", "histogram", "box"],
                value="bar",
                label="Chart type"
            )
            generate_btn = gr.Button("Generate Report")

            # outputs
            report_text = gr.Markdown()
            table = gr.DataFrame()
            download_link_output = gr.HTML()
            graph_plot = gr.Plot(visible=False)
            # wire the button and also wire checkbox/dropdown change events if you want them to auto-update
            generate_btn.click(
                fn=update_report,
                inputs=[show_graph_toggle, group_by_dropdown, chart_type_dropdown],
                outputs=[report_text, table, download_link_output, graph_plot]
            )
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
