from ml_template.tracking import git_state as gs


def test_returns_unknown_when_git_fails(monkeypatch):
    monkeypatch.setattr(gs, "_run_git", lambda *a: None)
    assert gs.git_sha() == "unknown"
    assert gs.git_is_dirty() is None
    assert gs.git_state() == {"git_sha": "unknown", "git_dirty": None}


def test_dirty_flag_from_porcelain_output(monkeypatch):
    monkeypatch.setattr(gs, "_run_git", lambda *a: " M src/x.py")
    assert gs.git_is_dirty() is True
    monkeypatch.setattr(gs, "_run_git", lambda *a: "")
    assert gs.git_is_dirty() is False


def test_short_sha_requests_short_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(gs, "_run_git", lambda *a: seen.append(a) or "abc")
    gs.git_sha(short=True)
    assert "--short" in seen[0]
