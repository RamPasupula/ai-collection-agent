# AI Collection Agent

An AI-powered collections assistant that helps call customers, manage reports, and visualize call statistics from **Twilio** and customer databases — all through an interactive **Gradio** dashboard and **FastAPI** backend.

---

## 🚀 Features

### 💬 AI Chat for Collections
- Chat with an AI collections agent in real-time.
- Integrated with **LangChain / LiteLLM** for intelligent responses.
- Configurable to connect with your customer database or CRM.

### 📊 Customer Report Dashboard
- Fetch customers to call from a database or API.
- Filter and group data by:
  - Best call day
  - Timezone
  - Missed payment status
  - Risk score
  - Amount due
- Download CSV reports directly.
- Visualize with:
  - **Bar charts**
  - **Pie charts**
  - **Histograms**
  - **Box plots**

### 📞 Twilio Call Reports
- Retrieve call logs from Twilio.
- View detailed call metrics: `status`, `duration`, `price`, `to`, `from`.
- Generate CSV downloads.
- Group and visualize calls by status, duration, cost, or time.
- Summarize call statistics with AI.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **FastAPI** – REST API backend
- **Gradio** – Interactive dashboard
- **Twilio Python SDK** – Call logs & telephony
- **LiteLLM / OpenAI GPT** – AI chat & summarization
- **Pandas** – Data processing
- **Plotly Express** – Charts & visualizations
- **Uvicorn** – ASGI server

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/RamPasupula/ai-collection-agent.git
cd ai-collection-agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Create .env file for your secrets
cp .env.example .env

# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# AI Provider (OpenAI / LiteLLM)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx



Access:
Gradio UI: http://localhost:7860

FastAPI Docs: http://localhost:8000/docs
