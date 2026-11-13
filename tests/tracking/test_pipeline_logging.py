import logging

import pytest

from ml_template.tracking import pipeline_logging as pl


@pytest.fixture(autouse=True)
def _isolated_logger():
    """Start each test with no handlers and restore whatever was there before.

    Other tests (and pytest's caplog) can leave handlers on the package logger,
    so these tests must not depend on, or wipe, that shared state.
    """
    base = logging.getLogger(pl.BASE_LOGGER_NAME)
    saved = (list(base.handlers), base.level, base.propagate)
    base.handlers = []
    yield base
    for handler in base.handlers:
        handler.close()
    base.handlers, base.level, base.propagate = saved[0], saved[1], saved[2]


def _file_targets(base):
    return [h.baseFilename for h in base.handlers if isinstance(h, logging.FileHandler)]


def test_app_log_receives_child_logger_records(tmp_path):
    app = tmp_path / "app.log"
    pl.setup_logging(app_logfile=app)
    logging.getLogger("ml_template.data.silver").info("hello silver")
    assert "hello silver" in app.read_text()


def test_setup_is_idempotent(tmp_path, _isolated_logger):
    app = tmp_path / "app.log"
    pl.setup_logging(app_logfile=app)
    after_first = list(_isolated_logger.handlers)
    pl.setup_logging(app_logfile=app)
    assert _isolated_logger.handlers == after_first
    assert len(_file_targets(_isolated_logger)) == 1
    assert sum(type(h) is logging.StreamHandler for h in after_first) == 1


def test_setup_ignores_capture_style_stream_handlers(tmp_path, _isolated_logger):
    class CaptureLike(logging.StreamHandler):
        pass

    _isolated_logger.addHandler(CaptureLike())
    pl.setup_logging(app_logfile=tmp_path / "app.log")
    assert sum(type(h) is logging.StreamHandler for h in _isolated_logger.handlers) == 1


def test_run_log_is_scoped_to_the_block(tmp_path, _isolated_logger):
    app, run1, run2 = tmp_path / "app.log", tmp_path / "r1.log", tmp_path / "r2.log"
    pl.setup_logging(app_logfile=app)
    before = list(_isolated_logger.handlers)
    log = logging.getLogger("ml_template.training.x")
    with pl.run_log(run1):
        log.info("one")
    with pl.run_log(run2):
        log.info("two")
    assert "one" in run1.read_text() and "two" not in run1.read_text()
    assert "two" in run2.read_text() and "one" not in run2.read_text()
    assert "one" in app.read_text() and "two" in app.read_text()
    assert _isolated_logger.handlers == before


def test_run_log_removed_even_on_error(tmp_path, _isolated_logger):
    pl.setup_logging(app_logfile=tmp_path / "app.log")
    before = list(_isolated_logger.handlers)
    with pytest.raises(RuntimeError), pl.run_log(tmp_path / "r.log"):
        raise RuntimeError("boom")
    assert _isolated_logger.handlers == before


def test_run_log_captures_info_without_setup_and_restores_level(tmp_path, _isolated_logger):
    _isolated_logger.setLevel(logging.NOTSET)  # a process that never called setup_logging
    run = tmp_path / "r.log"
    with pl.run_log(run):
        logging.getLogger("ml_template.training.x").info("captured")
        assert _isolated_logger.level == logging.INFO
    assert "captured" in run.read_text()
    assert _isolated_logger.level == logging.NOTSET
    assert _isolated_logger.handlers == []


def test_run_log_keeps_a_more_verbose_level(tmp_path, _isolated_logger):
    _isolated_logger.setLevel(logging.DEBUG)
    with pl.run_log(tmp_path / "r.log"):
        assert _isolated_logger.level == logging.DEBUG
    assert _isolated_logger.level == logging.DEBUG
