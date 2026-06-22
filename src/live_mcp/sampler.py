"""Live-state grounded task samplers."""

from __future__ import annotations

import random
from typing import Any, Protocol

from src.live_mcp.dependency_graph import ToolChain
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.query_generator import StructuredTask


class LiveStateSampler(Protocol):
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask: ...


class CalendarSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("calendar", session_id, "list_events", {})
        events = response["observation"]["events"]
        event = rng.choice(events)
        new_time = "2026-06-30T10:00"
        chain = ToolChain(
            chain_id="calendar:list_events->update_event",
            server_name="calendar",
            tools=["list_events", "update_event"],
            edges=[],
            difficulty=difficulty,
        )
        slots = {
            "event_id": event["event_id"],
            "title": event["title"],
            "old_time": event["start_time"],
            "new_time": new_time,
        }
        return StructuredTask(
            task_id="calendar_update_existing_event:template",
            server_name="calendar",
            tool_chain=chain,
            slots=slots,
            user_visible_slots=["title", "old_time", "new_time"],
            hidden_slots=["event_id"],
            success_criteria=[
                {
                    "type": "state_equals",
                    "server": "calendar",
                    "path": f"events.{event['event_id']}.start_time",
                    "value": new_time,
                }
            ],
            required_tools=["list_events", "update_event"],
            difficulty=difficulty,
        )


class ShoppingSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("shopping", session_id, "search_products", {"category": "keyboard", "max_price": 100})
        products = response["observation"]["products"]
        product = rng.choice(products)
        chain = ToolChain(
            chain_id="shopping:search_products->add_to_cart->checkout",
            server_name="shopping",
            tools=["search_products", "add_to_cart", "checkout"],
            edges=[],
            difficulty=difficulty,
        )
        slots = {
            "product_id": product["product_id"],
            "category": product["category"],
            "max_price": 100,
            "quantity": 1,
        }
        return StructuredTask(
            task_id="shopping_buy_product:template",
            server_name="shopping",
            tool_chain=chain,
            slots=slots,
            user_visible_slots=["category", "max_price"],
            hidden_slots=["product_id"],
            success_criteria=[
                {"type": "order_contains_product", "server": "shopping", "product_id": product["product_id"]},
                {"type": "cart_empty", "server": "shopping"},
            ],
            required_tools=["search_products", "add_to_cart", "checkout"],
            difficulty=difficulty,
        )
