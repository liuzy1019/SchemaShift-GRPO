"""Deterministic state seeders for MVP servers."""

from __future__ import annotations

import copy
import random
from typing import Any


class StateSeeder:
    def seed_state(self, server_name: str, session_id: str, seed: int) -> dict[str, Any]:
        if server_name == "calendar":
            return _calendar_state(seed)
        if server_name == "shopping":
            return _shopping_state(seed)
        raise ValueError(f"unsupported server: {server_name}")

    def reset_state(self, server_name: str, session_id: str, seed: int) -> dict[str, Any]:
        return copy.deepcopy(self.seed_state(server_name, session_id, seed))


def _calendar_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    titles = ["Team Sync", "Design Review", "Budget Check", "Customer Call"]
    events: dict[str, dict[str, Any]] = {}
    for idx, title in enumerate(titles, start=1):
        day = 22 + idx
        hour = 9 + ((idx + rng.randint(0, 3)) % 6)
        events[f"evt_{idx:03d}"] = {
            "event_id": f"evt_{idx:03d}",
            "title": title,
            "start_time": f"2026-06-{day:02d}T{hour:02d}:00",
            "end_time": f"2026-06-{day:02d}T{hour + 1:02d}:00",
            "attendees": ["alex@example.com", "sam@example.com"][: 1 + idx % 2],
        }
    return {"events": events, "next_event_num": len(events) + 1}


def _shopping_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    base = [
        ("prd_001", "K3 Keyboard", "keyboard", 79, 5),
        ("prd_002", "MX Mouse", "mouse", 49, 8),
        ("prd_003", "USB-C Hub", "hub", 35, 4),
        ("prd_004", "Noise Canceling Headphones", "audio", 99, 3),
    ]
    products = {
        pid: {
            "product_id": pid,
            "name": name,
            "category": category,
            "price": price + rng.randint(0, 5),
            "stock": stock,
        }
        for pid, name, category, price, stock in base
    }
    return {"products": products, "cart": [], "orders": {}, "next_order_num": 1}
