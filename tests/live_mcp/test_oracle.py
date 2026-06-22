import random

from src.live_mcp.oracle import OraclePlanner, OracleValidator
from src.live_mcp.sampler import CalendarSampler


def test_oracle_validates_on_fresh_reset(live_manager, executor):
    session = live_manager.create_session(seed=4)
    structured = CalendarSampler().sample_task(session.session_id, live_manager, "easy", random.Random(4))
    program = OraclePlanner().plan(structured)
    result = OracleValidator().validate(structured, program, live_manager, executor, seed=4)
    assert result.valid is True
    assert not result.failed_criteria
