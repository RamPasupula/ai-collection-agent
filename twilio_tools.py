from twilio.rest import Client
import os
import logging
# Replace with your Twilio SID/Auth Token (store securely)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
logger = logging.getLogger(__name__)


def call_customer(name: str, to_number: str, message: str) -> str:
    """Initiate a call to the customer using Twilio."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_FROM:
        return "Twilio credentials not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM."

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        twiml=f"<Response><Say>{message}</Say></Response>",
        to=to_number,
        from_=TWILIO_FROM,
    )
    return f"Call initiated to {name} at {to_number}. Call SID: {call.sid}"


def get_twilio_stats() -> list[dict]:
    """
    Returns Twilio call statistics as a list of dicts.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        calls = client.calls.list()
        #logger.info("MEta:: ",  vars(calls[0]))
        stats = []
        for call in calls:
            stats.append({
                "sid": call.sid,
                "status": call.status,
                "duration": int(call.duration or 0),
                "to": getattr(call, "to", None),
                "from": getattr(call, "from_", getattr(call, "from_formatted", None)),
                "start_time": call.start_time,
                "end_time": call.end_time,
                "price": call.price
            })
        return stats
    except Exception as e:
        logger.error(f"Error fetching Twilio stats: {e}")
        return []
