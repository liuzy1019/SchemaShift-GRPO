from src.live_mcp.state_seeder import StateSeeder


def test_state_seeder_is_deterministic_and_not_shared():
    seeder = StateSeeder()
    a = seeder.reset_state("calendar", "a", 7)
    b = seeder.reset_state("calendar", "b", 7)
    assert a == b
    a["events"]["evt_001"]["title"] = "Changed"
    assert b["events"]["evt_001"]["title"] != "Changed"
