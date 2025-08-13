import uuid
import asyncio
import inspect
from collection_agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import logging
# Create session service and runner globally to reuse
session_service = InMemorySessionService()
APP_NAME = "AI Collection Agent"
SESSION_ID = str(uuid.uuid4())
USER_ID = "Pasupula"
state_context = {"name": "Rama"}

# We'll create session and runner once on app startup
runner = None
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def init_agent():
    global runner
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=state_context,
    )
    runner = Runner(agent=root_agent, session_service=session_service, app_name=APP_NAME)
    
    
async def execute_tool(tool_func, **kwargs):
    """Run a tool asynchronously if possible."""
    logger.info(f"Executing tool: {tool_func.__name__} with args: {kwargs}")
    if asyncio.iscoroutinefunction(tool_func):
        return await tool_func(**kwargs)
    else:
        return tool_func(**kwargs)    
    

async def ask_agent(user_input: str) -> str:
    
    logger.info(f"User input: {user_input}")
    global runner
    if runner is None:
        raise RuntimeError("Agent runner not initialized")

    user_query = types.Content(role="user", parts=[types.Part(text=user_input)])
    response_text = None

    # Run the sync generator in a thread
    def run_sync_agent():
        for event in runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_query,
        ):
            yield event

    # Iterate in async context
    for event in await asyncio.to_thread(lambda: list(run_sync_agent())):
        if getattr(event, "tool_calls", None):
            for tool_call in event.tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.arguments or {}
                tool_func = next((t for t in root_agent.tools if t.__name__ == tool_name), None)
                if tool_func:
                    try:
                        logger.info(f"Running tool: {tool_name} with args: {tool_args}")
                        result = await execute_tool(tool_func, **tool_args)
                    except Exception as e:
                        result = f"Error running tool {tool_name}: {e}"

                    await runner.send_tool_result(
                        user_id=USER_ID,
                        session_id=SESSION_ID,
                        tool_call_id=tool_call.id,
                        content=str(result),
                    )
                else:
                    logger.error(f"Tool {tool_name} not found in agent tools")    

        elif event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    return response_text or event.content
