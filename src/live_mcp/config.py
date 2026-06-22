"""YAML config loading for Live MCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServerConfig:
    name: str
    type: str
    enabled: bool
    command: dict[str, Any]
    transport: dict[str, Any]
    session: dict[str, Any]
    tools: dict[str, Any]
    sampler: dict[str, Any] = field(default_factory=dict)
    reward_profile: str = ""
    dependency_graph: dict[str, Any] = field(default_factory=dict)
    query_templates: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SuiteConfig:
    suite_name: str
    servers: list[ServerConfig]
    rollout: dict[str, Any]
    reward: dict[str, Any]
    trace: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=lambda: {"backend": "replay"})
    live: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def load_server_config(path: str | Path, root: Path | None = None) -> ServerConfig:
    root = root or project_root()
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = root / full_path
    data = _load_yaml(full_path)
    return ServerConfig(**data)


def load_suite_config(path: str | Path, root: Path | None = None) -> SuiteConfig:
    root = root or project_root()
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = root / full_path
    data = _load_yaml(full_path)
    server_paths = data.get("servers", [])
    servers = [load_server_config(server_path, root=root) for server_path in server_paths]
    return SuiteConfig(
        suite_name=data["suite_name"],
        servers=servers,
        rollout=data.get("rollout", {}),
        reward=data.get("reward", {}),
        trace=data.get("trace", {}),
        environment=data.get("environment", {"backend": "replay"}),
        live=data.get("live", {}),
        replay=data.get("replay", {}),
    )
