"""Stateful shopping server for Live MCP smoke tests."""

from __future__ import annotations

from typing import Any

from src.live_mcp.server_base import StatefulToolServer, _result, serve


TOOLS = [
    {
        "name": "search_products",
        "description": "Search products by query, category, and price.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "max_price": {"type": "number"},
            },
            "required": [],
        },
        "annotations": {"readonly": True, "mutating": False},
    },
    {
        "name": "add_to_cart",
        "description": "Add an in-stock product to the cart.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["product_id", "quantity"],
        },
        "annotations": {"mutating": True},
    },
    {
        "name": "remove_from_cart",
        "description": "Remove a product from the cart.",
        "input_schema": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
        "annotations": {"mutating": True},
    },
    {
        "name": "checkout",
        "description": "Checkout all cart items.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"mutating": True},
    },
    {
        "name": "get_order",
        "description": "Get an order by id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        "annotations": {"readonly": True, "mutating": False},
    },
]


class ShoppingServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("shopping", TOOLS)
        self.handlers = {
            "search_products": self.search_products,
            "add_to_cart": self.add_to_cart,
            "remove_from_cart": self.remove_from_cart,
            "checkout": self.checkout,
            "get_order": self.get_order,
        }

    def search_products(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        query = str(arguments.get("query", "")).lower()
        category = arguments.get("category")
        max_price = arguments.get("max_price")
        products = []
        for product in state["products"].values():
            if category and product["category"] != category:
                continue
            if query and query not in product["name"].lower() and query not in product["category"]:
                continue
            if max_price is not None and product["price"] > float(max_price):
                continue
            products.append(product)
        return _result(True, {"products": products}, None, "", False)

    def add_to_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        product_id = arguments["product_id"]
        quantity = int(arguments["quantity"])
        product = state["products"].get(product_id)
        if product is None:
            raise KeyError(f"product not found: {product_id}")
        if product["stock"] < quantity:
            raise KeyError(f"insufficient stock: {product_id}")
        product["stock"] -= quantity
        state["cart"].append({"product_id": product_id, "quantity": quantity, "unit_price": product["price"]})
        return _result(True, {"cart": list(state["cart"])}, None, "", True)

    def remove_from_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        product_id = arguments["product_id"]
        kept = []
        removed = []
        for item in state["cart"]:
            if item["product_id"] == product_id:
                removed.append(item)
                state["products"][product_id]["stock"] += item["quantity"]
            else:
                kept.append(item)
        state["cart"] = kept
        return _result(True, {"removed": removed, "cart": list(state["cart"])}, None, "", bool(removed))

    def checkout(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        if not state["cart"]:
            raise KeyError("cart is empty")
        order_id = f"ord_{state['next_order_num']:03d}"
        state["next_order_num"] += 1
        order = {
            "order_id": order_id,
            "items": list(state["cart"]),
            "total": sum(item["quantity"] * item["unit_price"] for item in state["cart"]),
        }
        state["orders"][order_id] = order
        state["cart"] = []
        return _result(True, {"order": order}, None, "", True)

    def get_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        order_id = arguments["order_id"]
        if order_id not in state["orders"]:
            raise KeyError(f"order not found: {order_id}")
        return _result(True, {"order": state["orders"][order_id]}, None, "", False)


if __name__ == "__main__":
    serve(ShoppingServer())
