# Agentic Food Delivery Chatbot (FastAPI + LangGraph)

A small assignment-ready chatbot that uses an LLM to interpret natural-language requests and call Python tools. All menu and order data is mocked locally. No payment, restaurant, or delivery APIs are used.

## Features
- Browse the mock menu, optionally by category.
- Place an order after item and quantity are known.
- Check a delivery status by order ID.
- Submit 1–5 star feedback for an existing order.
- Preserve conversation context using LangGraph's in-memory checkpointer and a `thread_id`.
- Handle missing details conversationally, invalid order IDs, unavailable items, and invalid quantities/ratings.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then export the variables. The app does not automatically load `.env`; optionally install `python-dotenv` and add `from dotenv import load_dotenv; load_dotenv()` at the top of `app.py`, or set environment variables in your terminal.

PowerShell:
```powershell
$env:OPENAI_API_KEY="your_key"
uvicorn app:app --reload
```

macOS/Linux:
```bash
export OPENAI_API_KEY="your_key"
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

## API

`POST /chat`
```json
{
  "thread_id": "customer-123",
  "message": "Show me pizzas"
}
```
Response:
```json
{
  "thread_id": "customer-123",
  "reply": "Here are the available pizzas..."
}
```

Use the same `thread_id` across turns for one conversation. Use a different ID for a separate customer/session.

## Four+ test scenarios

1. **Menu browsing**: `Show me the menu` or `Show me pizzas`. The assistant calls `browse_menu`.
2. **Place an order**: `Order 2 masala dosa`. The assistant calls `place_order` and returns a generated mock order ID and total.
3. **Missing information / context**: `I want to order food` should prompt for an item; reply `a veg burger` in the same `thread_id`. The model should use the follow-up context and ask for quantity if needed.
4. **Order status**: `Check order FD1001`. The assistant calls `check_order_status`; seeded status is `Out for delivery`.
5. **Invalid order ID**: `Where is order FD9999?` The tool returns a not-found error, which the assistant explains without inventing a status.
6. **Unavailable item**: `Order a chocolate brownie`. The tool reports it is unavailable and suggests choosing another item.
7. **Feedback**: `Give 5 stars to FD1001; great delivery`. The assistant calls `submit_feedback`.

These are reproducible manual scenarios through Swagger at `/docs`. Since the LLM controls natural-language routing, wording can affect the exact phrasing of the response.

## Architecture

```text
Client
  |
  v
FastAPI POST /chat
  |
  v
LangGraph StateGraph (MessagesState + thread_id checkpoint)
  |
  v
LLM assistant node -- tool call? --> ToolNode
  ^                                  |
  |____________ tool result __________|
  |
  v
Natural-language response
```

### Design choices
- **LangGraph** manages the agent loop: the model decides whether to answer directly or invoke one of the registered tools; tool results are fed back to the model before it responds.
- **Python tools** own the actual operations. The LLM does not create order records by merely saying it did.
- **Mock data** makes the assignment self-contained and avoids external APIs.
- **In-memory checkpointing** supports context across turns for a `thread_id`. It resets when the process restarts and is not suitable for multi-worker production deployment.
- **FastAPI + Pydantic** provide a simple typed HTTP contract and request validation.

### Error handling
- Unknown item: tool returns a clear error and asks the assistant to suggest menu options.
- Unavailable item: order is rejected; no order is created.
- Invalid quantity: constrained to 1–20.
- Unknown order ID: status/feedback tools return a not-found result without crashing.
- Invalid rating: only 1–5 accepted.
- Missing API key: app startup reports a configuration error.
- Runtime exception: API returns a generic error instead of exposing a traceback or secrets.

### Limitations
The in-memory dictionaries and checkpoint are intentionally for a demo. Restarting the app clears newly created orders and feedback. For production, add persistence, authentication, idempotency, structured logging, and stronger order confirmation rules. Never use an LLM-only check as authorization for a real payment or purchase.
