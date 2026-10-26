
import os
from typing import TypedDict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv

load_dotenv()



# ============================================================
# 1. MOCK MENU WITH INVENTORY QUANTITIES
# ============================================================

MENU = {
    "margherita pizza": {
        "price": 299,
        "category": "pizza",
        "stock": 10
    },
    "farmhouse pizza": {
        "price": 399,
        "category": "pizza",
        "stock": 10
    },
    "veg burger": {
        "price": 149,
        "category": "burger",
        "stock": 10
    },
    "chicken burger": {
        "price": 199,
        "category": "burger",
        "stock": 10
    },
    "masala dosa": {
        "price": 99,
        "category": "south indian",
        "stock": 10
    },
    "paneer wrap": {
        "price": 179,
        "category": "wrap",
        "stock": 10
    },
    "cold coffee": {
        "price": 89,
        "category": "beverage",
        "stock": 10
    },
    "chocolate brownie": {
        "price": 119,
        "category": "dessert",
        "stock": 0
    }
}


# ============================================================
# 2. MOCK ORDER DATABASE
# ============================================================

ORDERS = {
    "FD1001": {
        "items": [
            {"item": "masala dosa", "quantity": 2}
        ],
        "total": 198,
        "status": "Out for delivery",
        "feedback": []
    },
    "FD1002": {
        "items": [
            {"item": "veg burger", "quantity": 1}
        ],
        "total": 149,
        "status": "Preparing",
        "feedback": []
    }
}


# ============================================================
# 3. TOOL: BROWSE MENU
# ============================================================

@tool
def browse_menu(category: str = "") -> dict:
    """Show menu items, prices, categories, and current remaining stock.
    Optionally filter by category such as pizza, burger, or beverage.
    """

    items = {}

    for name, data in MENU.items():
        if (
            category
            and category.lower() not in data["category"].lower()
        ):
            continue

        items[name] = {
            "price_inr": data["price"],
            "category": data["category"],
            "quantity_available": data["stock"],
            "available": data["stock"] > 0
        }

    return {
        "ok": True,
        "items": items
    }


# ============================================================
# 4. TOOL: PLACE ORDER AND UPDATE INVENTORY
# ============================================================

@tool
def place_order(item: str, quantity: int = 1) -> dict:
    """Place an order for a menu item.
    Validate item availability and quantity.
    Reduce inventory only after a valid order is created.
    """

    normalized = item.strip().lower()

    # Validate quantity
    if quantity < 1 or quantity > 20:
        return {
            "ok": False,
            "error": "Quantity must be between 1 and 20."
        }

    # Validate menu item
    if normalized not in MENU:
        return {
            "ok": False,
            "error": (
                f"'{item}' is not on the menu. "
                "Please browse the menu and choose an item."
            )
        }

    menu_item = MENU[normalized]
    available_stock = menu_item["stock"]

    # Validate inventory
    if available_stock < quantity:
        return {
            "ok": False,
            "error": (
                f"Only {available_stock} unit(s) of "
                f"{normalized} are available. "
                f"You requested {quantity}. "
                "Please reduce the quantity or choose another item."
            ),
            "quantity_available": available_stock
        }

    # Calculate total
    total = menu_item["price"] * quantity

    # Generate a unique mock order ID
    order_id = "FD" + uuid4().hex[:6].upper()

    # Create order record
    ORDERS[order_id] = {
        "items": [
            {
                "item": normalized,
                "quantity": quantity
            }
        ],
        "total": total,
        "status": "Received",
        "feedback": []
    }

    # Reduce inventory after successful validation
    menu_item["stock"] -= quantity

    return {
        "ok": True,
        "message": "Order placed successfully!",
        "order_id": order_id,
        "items": ORDERS[order_id]["items"],
        "total_inr": total,
        "status": "Received",
        "remaining_stock": menu_item["stock"]
    }


# ============================================================
# 5. TOOL: CHECK ORDER STATUS
# ============================================================

@tool
def check_order_status(order_id: str) -> dict:
    """Check the delivery status of an order using its order ID."""

    key = order_id.strip().upper()

    if key not in ORDERS:
        return {
            "ok": False,
            "error": (
                f"Order ID '{order_id}' was not found. "
                "Please check the order ID and try again."
            )
        }

    order = ORDERS[key]

    return {
        "ok": True,
        "order_id": key,
        "status": order["status"],
        "items": order["items"],
        "total_inr": order["total"]
    }


# ============================================================
# 6. TOOL: SUBMIT FEEDBACK
# ============================================================

@tool
def submit_feedback(
    order_id: str,
    rating: int,
    comment: str = ""
) -> dict:
    """Submit a 1-5 star rating and optional comment for an order."""

    key = order_id.strip().upper()

    if key not in ORDERS:
        return {
            "ok": False,
            "error": f"Order ID '{order_id}' was not found."
        }

    if rating < 1 or rating > 5:
        return {
            "ok": False,
            "error": "Rating must be between 1 and 5."
        }

    feedback = {
        "rating": rating,
        "comment": comment.strip()
    }

    ORDERS[key]["feedback"].append(feedback)

    return {
        "ok": True,
        "message": "Thank you! Your feedback has been recorded.",
        "order_id": key,
        "feedback": feedback
    }


# ============================================================
# 7. LANGGRAPH AGENT
# ============================================================

TOOLS = [
    browse_menu,
    place_order,
    check_order_status,
    submit_feedback
]


def build_graph():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it before starting the app."
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0
    )

    model = llm.bind_tools(TOOLS)

    system_prompt = """
You are a helpful food delivery chatbot.

Use Python tools for menu, order, status, and feedback operations.

IMPORTANT RULES:
1. When the user asks to show or browse the menu,
   call browse_menu.
2. Print every returned menu item, its price, and its
   current available quantity.
3. Do not replace the menu with "Here is the menu..."
   or omit the menu items.
4. When filtering by category, show all returned items
   in that category.
5. For orders, call place_order.
6. Never claim an order succeeded unless the tool
   returns ok=true.
7. Explain insufficient stock and invalid order IDs.
8. Use conversation history for follow-up questions.
9. Currency is INR.
"""

    def assistant_node(state: MessagesState):
        messages = [
            {"role": "system", "content": system_prompt}
        ] + state["messages"]

        response = model.invoke(messages)

        return {
            "messages": [response]
        }

    graph = StateGraph(MessagesState)

    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "assistant")

    graph.add_conditional_edges(
        "assistant",
        tools_condition
    )

    graph.add_edge("tools", "assistant")

    return graph.compile(
        checkpointer=InMemorySaver()
    )


# ============================================================
# 8. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic Food Delivery Chatbot",
    description=(
        "Food delivery chatbot using FastAPI, LangGraph, "
        "LLM tool calling, and dynamic menu inventory."
    ),
    version="2.0.0"
)

agent = None


# ============================================================
# 9. REQUEST AND RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(
        default="demo-user",
        min_length=1,
        max_length=100
    )


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


# ============================================================
# 10. INITIALIZE AGENT
# ============================================================

@app.on_event("startup")
def startup():
    global agent
    agent = build_graph()


# ============================================================
# 11. HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Food Delivery Agent API is running",
        "docs": "/docs",
        "chat_endpoint": "POST /chat"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_initialized": agent is not None
    }


# ============================================================
# 12. MENU INVENTORY ENDPOINT
# ============================================================

@app.get("/menu")
def get_menu():
    """Return the current menu and live remaining stock."""

    return {
        name: {
            "price_inr": data["price"],
            "category": data["category"],
            "quantity_available": data["stock"],
            "available": data["stock"] > 0
        }
        for name, data in MENU.items()
    }


# ============================================================
# 13. CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent is not initialized. Check API configuration."
        )

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=request.message)
                ]
            },
            config=config
        )

        last_message = result["messages"][-1]
        reply = last_message.content

        if isinstance(reply, list):
            reply = " ".join(
                part.get("text", "")
                for part in reply
                if isinstance(part, dict)
            )

        return ChatResponse(
            thread_id=request.thread_id,
            reply=str(reply)
        )

    except Exception as exc:
        # Log the exception in a real application.
        print(f"Chat processing error: {type(exc).__name__}")

        raise HTTPException(
            status_code=500,
            detail="Chat processing failed. Please try again."
        ) from exc
