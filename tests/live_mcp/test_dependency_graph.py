from src.live_mcp.dependency_graph import DependencyGraphBuilder, DependencyEdge


def test_dependency_graph_extracts_chains():
    builder = DependencyGraphBuilder()
    graph = builder.build(
        "shopping",
        [{"name": "search_products"}, {"name": "add_to_cart"}, {"name": "checkout"}],
        [
            DependencyEdge("search_products", "add_to_cart", "explicit"),
            DependencyEdge("add_to_cart", "checkout", "implicit"),
        ],
    )
    chains = builder.extract_chains(graph, min_len=2, max_len=5)
    assert any(chain.tools == ["search_products", "add_to_cart", "checkout"] for chain in chains)
