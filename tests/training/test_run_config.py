from pathlib import Path

import pytest
from pydantic import ValidationError

from ml_template.paths import CONFIGS
from ml_template.training import run_config as rc

MODEL = "type: xgb\nparams:\n  max_depth: 4\n  learning_rate: 0.05\n"


def _write(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "xgb.yaml").write_text(MODEL)
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment_name: t1\n"
        "data:\n  dataset_name: ds\n  dataset_config: c.yaml\n"
        "  silver_dataset_version_id: silver_x\n"
        f"model_file: {tmp_path / 'xgb.yaml'}\n" + extra
    )
    return exp


def test_load_defaults_and_override(tmp_path):
    cfg = rc.load_config(_write(tmp_path), ["model.params.max_depth=7", "seed=1"])
    assert cfg.model.params.max_depth == 7
    assert cfg.seed == 1
    assert cfg.schema_version == 2
    p = cfg.model.params
    assert p.n_estimators == 2000
    assert cfg.analysis.error_sample_size == 25
    assert (p.early_stopping_rounds, p.scale_pos_weight, p.device) == (50, "auto", "cpu")
    assert cfg.training.validation_fraction == 0.2
    assert cfg.analysis.permutation_importance and cfg.analysis.permutation_repeats == 10


def test_standard_metrics_are_always_present(tmp_path):
    exp = _write(tmp_path)
    cfg = rc.load_config(exp)
    assert cfg.metrics.names[:5] == [
        "pr_auc",
        "roc_auc",
        "brier",
        "recall_at_alert_rate",
        "precision_at_alert_rate",
    ]
    assert cfg.metrics.alert_rates == [0.05, 0.10]
    custom = rc.load_config(exp, ["metrics.names=[roc_auc]", "metrics.alert_rates=[0.01]"])
    assert "pr_auc" in custom.metrics.names  # cannot be removed
    assert custom.metrics.alert_rates == [0.01, 0.05, 0.10]  # extras are added


def test_scale_pos_weight_accepts_auto_or_positive_number(tmp_path):
    exp = _write(tmp_path)
    assert (
        rc.load_config(exp, ["model.params.scale_pos_weight=10"]).model.params.scale_pos_weight
        == 10
    )
    auto = rc.load_config(exp, ["model.params.scale_pos_weight=auto"])
    assert auto.model.params.scale_pos_weight == "auto"
    for bad in ("0", "-1", "none"):
        with pytest.raises(ValidationError):
            rc.load_config(exp, [f"model.params.scale_pos_weight={bad}"])


def test_invalid_values_rejected(tmp_path):
    exp = _write(tmp_path)
    for override in (
        "model.params.max_dept=3",  # unknown key
        "model.params.learning_rate=-1",
        "model.params.objective=reg:squarederror",
        "model.params.early_stopping_rounds=null",  # early stopping is required
        "model.params.early_stopping_rounds=0",
        "model.params.device=tpu",
        "training.validation_fraction=0.6",
        "training.validation_fraction=0",
        "training.refit_rounds_factor=5",
        "experiment_name=bad name",
        "metrics.primary=f1",  # not a known result key
        "metrics.names=[nope]",
        "metrics.alert_rates=[0, 0.1]",
        "split.holdout_fraction=1.5",
        "analysis.permutation_repeats=0",
        "analysis.error_sample_size=-1",
        "analysis.unknown=1",
        "schema_version=1",
    ):
        with pytest.raises(ValidationError):
            rc.load_config(exp, [override])


def test_primary_may_be_alert_key(tmp_path):
    cfg = rc.load_config(_write(tmp_path), ["metrics.primary=recall_at_5pct"])
    assert cfg.metrics.primary == "recall_at_5pct"


def test_override_syntax_error(tmp_path):
    with pytest.raises(ValueError, match="key=value"):
        rc.load_config(_write(tmp_path), ["seed"])


def test_hash_stable_sensitive_and_ignores_analysis(tmp_path):
    exp = _write(tmp_path)
    a, b = rc.load_config(exp), rc.load_config(exp)
    c = rc.load_config(exp, ["training.validation_fraction=0.25"])
    d = rc.load_config(exp, ["analysis.permutation_repeats=3", "analysis.error_sample_size=0"])
    e = rc.load_config(exp, ["metrics.alert_rates=[0.01]"])
    assert rc.config_hash(a) == rc.config_hash(b) == rc.config_hash(d)
    assert rc.config_hash(a) != rc.config_hash(c)
    assert rc.config_hash(a) != rc.config_hash(e)  # extra metrics do change the config


def test_flatten_params(tmp_path):
    flat = rc.flatten_params(rc.load_config(_write(tmp_path)))
    assert flat["model.params.max_depth"] == "4"
    assert flat["model.params.scale_pos_weight"] == "auto"
    assert flat["metrics.alert_rates"] == "[0.05, 0.1]"
    assert flat["training.refit_rounds_factor"] == "1.0"
    assert flat["analysis.permutation_repeats"] == "10"


def test_expand_sweep(tmp_path):
    exp = _write(tmp_path)
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        f"base: {exp}\nexperiment_name: grid1\ngrid:\n"
        "  model.params.max_depth: [3, 5]\n"
        "  model.params.scale_pos_weight: [auto, 10]\n"
    )
    cfgs = rc.expand_sweep(sweep)
    assert len(cfgs) == 4
    assert {c.experiment_name for c in cfgs} == {"grid1"}
    assert len({rc.config_hash(c) for c in cfgs}) == 4


def test_sweep_guards(tmp_path):
    exp = _write(tmp_path)
    big = tmp_path / "big.yaml"
    values = list(range(1, 30))
    big.write_text(
        f"base: {exp}\ngrid:\n  model.params.max_depth: {values}\n"
        f"  model.params.n_estimators: {values}\n"
    )
    with pytest.raises(ValueError, match="limit"):
        rc.expand_sweep(big)
    typo = tmp_path / "typo.yaml"
    typo.write_text(f"base: {exp}\ngrid:\n  seed: [1]\nbasee: x\n")
    with pytest.raises(ValueError, match="unknown sweep keys"):
        rc.expand_sweep(typo)
    empty = tmp_path / "empty.yaml"
    empty.write_text(f"base: {exp}\ngrid:\n  seed: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        rc.expand_sweep(empty)


def test_resolve_path_falls_back_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert rc.resolve_path("configs/model/xgb.yaml") == CONFIGS / "model" / "xgb.yaml"


def test_repository_configs_are_valid():
    cfg = rc.load_config(CONFIGS / "experiment" / "xgb_baseline.yaml")

    assert cfg.task == "binary_classification"
    assert cfg.model.type == "xgb"
    assert cfg.model.params.objective == "binary:logistic"
    assert cfg.model.params.eval_metric == "aucpr"
    assert cfg.split.method == "stratified"
    assert cfg.metrics.primary == "pr_auc"

    # Examples may choose an explicit budget rather than the schema default.
    assert cfg.model.params.n_estimators > 0
    assert (
        0
        < cfg.model.params.early_stopping_rounds
        <= cfg.model.params.n_estimators
    )
    assert cfg.analysis.error_sample_size >= 0

    sweep = rc.expand_sweep(CONFIGS / "sweep" / "xgb_grid.yaml")
    assert len(sweep) == 12
    assert len({rc.config_hash(item) for item in sweep}) == 12
    assert all(item.task == "binary_classification" for item in sweep)
    assert all(item.model.type == "xgb" for item in sweep)


def test_cli_prints_hash(tmp_path, capsys):
    rc.main(["--config", str(_write(tmp_path)), "--set", "seed=3"])
    out = capsys.readouterr().out
    assert '"seed": 3' in out and "hash:" in out
