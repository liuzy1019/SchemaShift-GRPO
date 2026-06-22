from __future__ import annotations

import pytest

from src.live_mcp.config import load_suite_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager


@pytest.fixture()
def suite():
    return load_suite_config("configs/live_mcp/suite_mvp.yaml")


@pytest.fixture()
def live_manager(suite):
    manager = LiveMCPManager(suite)
    manager.start_suite()
    try:
        yield manager
    finally:
        manager.stop_suite()


@pytest.fixture()
def executor(live_manager):
    return LiveMCPExecutor(live_manager, live_manager.registry)
