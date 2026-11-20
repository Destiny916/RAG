import hashlib
import json
from datetime import date

import pytest

from local_rag import intake as hook


@pytest.fixture
def event(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "ROOT", tmp_path)
    (tmp_path / "knowledge" / "alpha").mkdir(parents=True)
    return {
        "hook_event_name": "Stop", "stop_hook_active": False, "cwd": str(tmp_path),
        "session_id": "session-one", "turn_id": "turn-one",
        "last_assistant_message": "UNSAVED_TEST_TRANSCRIPT",
    }


def request(event):
    result = hook.request_review(event)
    assert result["decision"] == "block"
    assert "UNSAVED_TEST_TRANSCRIPT" not in result["reason"]
    token = hashlib.sha256(json.dumps([event["session_id"], event["turn_id"]]).encode()).hexdigest()
    assert token in result["reason"]
    return token


def enable_stop():
    scope_file = hook.ROOT / ".rag" / "stop-workspaces.json"
    scope_file.parent.mkdir()
    scope_file.write_text(json.dumps([str(hook.ROOT)]))


def test_stop_requests_only_one_review_per_turn(event):
    assert hook.request_review(event) is None
    assert not (hook.ROOT / ".rag").exists()
    enable_stop()
    assert hook.request_review({**event, "stop_hook_active": True}) is None
    assert hook.request_review({**event, "hook_event_name": "UserPromptSubmit"}) is None
    assert hook.request_review({**event, "cwd": "relative"}) is None
    first = request(event)
    assert hook.request_review(event) is None
    other = {**event, "cwd": str(hook.ROOT.parent), "turn_id": "turn-two"}
    assert hook.request_review(other) is None
    assert request({**other, "cwd": str(hook.ROOT / "knowledge")}) != first
    assert request({**event, "session_id": "session-two"}) != first


def test_review_draft_is_private_and_does_not_change_knowledge(event):
    enable_stop()
    token = request(event)
    target = "alpha/new.md"
    draft = {
        "action": "new_file", "kind": "guide", "target": target, "summary": "Verified example",
        "source": "src/settings.py:4", "verified_on": date.today().isoformat(), "scope": "alpha",
    }
    with pytest.raises(ValueError):
        hook.record(token, {**draft, "target": "../outside.md"})
    with pytest.raises(ValueError):
        hook.record(token, {**draft, "summary": "api_key: example-test-credential"})
    with pytest.raises(ValueError):
        hook.record("0" * 64, draft)
    hook.record(token, draft)
    stored = hook.review_path(token) / "decision.json"
    assert json.loads(stored.read_text()) == draft
    assert stored.stat().st_mode & 0o077 == 0
    assert not (hook.ROOT / "knowledge" / target).exists()
    with pytest.raises(FileExistsError):
        hook.record(token, {"action": "skip", "reason": "duplicate"})
    assert json.loads(stored.read_text()) == draft
