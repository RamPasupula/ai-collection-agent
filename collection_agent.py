from collection_tools import get_customer_info, list_defaulted_customers, call_customer_by_name, get_customers_to_call, default_customers_fallback
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import os
import openai
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]

# from twilio_tools import call_customer
root_agent = Agent(
    name="loan_process_agent",
    model=LiteLlm(model="openai/gpt-4.1"),
    description="Agent to assist with personal loans and retrieving customer records.",
    instruction=(
        """
        You are a Personal Loan Assistant Agent.

        - If a user request matches a specific tool, call that tool.
        - If no tool clearly matches, CALL `default_customers_fallback` with the user prompt.
        - Use the tool's output directly in your answer (it already analyzes the data).
        - Never invent data not present in the database.
        """
    ),
    tools=[
        get_customer_info,
        call_customer_by_name,
        get_customers_to_call,
       # list_defaulted_customers,
        default_customers_fallback,  # <-- fallback tool
    ],
    # tools=[get_customer_info, call_customer_by_name, get_customers_to_call,  list_defaulted_customers , default_tool],
)
