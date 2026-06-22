"""Configuration-backed tool dependency graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DependencyEdge:
    source_tool: str
    target_tool: str
    relation: str
    source_output_path: str = ""
    target_argument_path: str = ""
    description: str = ""


@dataclass
class ToolChain:
    chain_id: str
    server_name: str
    tools: list[str]
    edges: list[DependencyEdge]
    difficulty: str


@dataclass
class ToolDependencyGraph:
    server_name: str
    tools: list[str]
    edges: list[DependencyEdge] = field(default_factory=list)


class DependencyGraphBuilder:
    def build(
        self,
        server_name: str,
        tool_schemas: list[dict[str, Any]],
        config_edges: list[DependencyEdge],
    ) -> ToolDependencyGraph:
        tools = [schema["name"] for schema in tool_schemas if "name" in schema]
        return ToolDependencyGraph(server_name=server_name, tools=tools, edges=config_edges)

    def extract_chains(
        self,
        graph: ToolDependencyGraph,
        min_len: int = 2,
        max_len: int = 5,
    ) -> list[ToolChain]:
        adjacency: dict[str, list[DependencyEdge]] = {}
        for edge in graph.edges:
            adjacency.setdefault(edge.source_tool, []).append(edge)
        chains: list[ToolChain] = []
        for start in adjacency:
            self._dfs(graph, start, [], [], min_len, max_len, chains)
        unique: dict[tuple[str, ...], ToolChain] = {}
        for chain in chains:
            unique.setdefault(tuple(chain.tools), chain)
        return list(unique.values())

    def _dfs(
        self,
        graph: ToolDependencyGraph,
        current: str,
        path: list[str],
        edges: list[DependencyEdge],
        min_len: int,
        max_len: int,
        chains: list[ToolChain],
    ) -> None:
        path = path + [current]
        if min_len <= len(path) <= max_len:
            chains.append(
                ToolChain(
                    chain_id=f"{graph.server_name}:{'->'.join(path)}",
                    server_name=graph.server_name,
                    tools=path,
                    edges=list(edges),
                    difficulty="easy" if len(path) <= 2 else "medium",
                )
            )
        if len(path) >= max_len:
            return
        for edge in graph.edges:
            if edge.source_tool == current and edge.target_tool not in path:
                self._dfs(
                    graph,
                    edge.target_tool,
                    path,
                    edges + [edge],
                    min_len,
                    max_len,
                    chains,
                )


def edges_from_config(raw_edges: list[dict[str, Any]]) -> list[DependencyEdge]:
    return [DependencyEdge(**edge) for edge in raw_edges]
