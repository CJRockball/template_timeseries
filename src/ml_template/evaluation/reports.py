"""Per-batch scorecard: did the queue work, in numbers, as an artifact.

Reads a score run (scores.parquet + config.json), joins true labels when they
exist (simulated demo batches and the holdout replay carry them; genuinely new
lots would not), rebuilds the window's engineered flags for the mechanism
check, and writes scorecard.json into the same folder. One registry row per
scorecard: type='scorecard', parent_run=<score run id>.

Metric definitions match training by construction — capture uses the same
recall_at_flagrate as evaluation.py, so serving and evaluation numbers are
definitionally identical.

Entry point: project-scorecard = project.scorecard:main
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.data import schema
from ml_template.data.extract import extract
from ml_template.db.connection import get_engine
from ml_template.evaluation.metrics import recall_at_flagrate
from ml_template.features.build import build_features
from ml_template.paths import LOGS
from ml_template.scoring.batch import SCORES, append_index, git_sha
from ml_template.tracking.utils import setup_logging

logger = logging.getLogger(__name__)

FLAG_RATES = (0.05, 0.10)  # fixed grid — comparable across batches without run archaeology


def resolve_score_run(run: str, label: str | None) -> Path:
    """Score run dir by id, or the latest one (optionally of one label)."""
    if run != "latest":
        d = SCORES / run
        if not d.is_dir():
            raise FileNotFoundError(f"no score run {run} in {SCORES}")
        return d
    candidates = sorted(
        p for p in SCORES.iterdir() if p.is_dir() and (p / "scores.parquet").exists()
    )
    if label:
        candidates = [p for p in candidates if p.name.endswith(f"__{label}")]
    if not candidates:
        raise FileNotFoundError(f"no score runs in {SCORES} (label={label!r})")
    return candidates[-1]


def fetch_labels(engine, wafer_ids: np.ndarray) -> pd.DataFrame | None:
    """True labels for the scored wafers; None when the lots are unlabeled."""
    marks = ",".join("?" * len(wafer_ids))
    df = pd.read_sql(
        f"SELECT {schema.KEY_COL}, {schema.TARGET_COL} FROM wafer_labels "
        f"WHERE {schema.KEY_COL} IN ({marks})",
        engine,
        params=tuple(int(i) for i in wafer_ids),
    )
    return df if not df.empty else None


def fetch_flags(engine, cfg: dict) -> pd.DataFrame:
    """f_miss_clq14 per wafer, rebuilt through the same path the scorer used."""
    frame = extract(engine, start=cfg["start"], end=cfg["end"], cutoff=None, exclude_after=None)
    engineered, _ = build_features(frame)
    return engineered[[schema.KEY_COL, "f_miss_clq14"]]


def compute_panel(
    scores: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    flags: pd.DataFrame | None = None,
    flag_rates: tuple[float, ...] = FLAG_RATES,
) -> dict:
    """The scorecard panel: distribution, mechanism, capture. Pure function."""
    panel: dict = {"n_scored": int(len(scores))}
    p = scores["p_cal"].fillna(scores["score_raw"])
    panel["p_median"] = float(p.median())
    panel["p_mean"] = float(p.mean())
    panel["p90"] = float(p.quantile(0.9))

    if flags is not None:  # mechanism: clique-14 enrichment in the top decile
        f = scores[[schema.KEY_COL, "decile"]].merge(flags, on=schema.KEY_COL, how="left")
        top_share = float(f.loc[f["decile"] == 1, "f_miss_clq14"].mean())
        batch_share = float(f["f_miss_clq14"].mean())
        panel["clq14_top_decile_share"] = top_share
        panel["clq14_batch_share"] = batch_share
        panel["clq14_enrichment"] = top_share / batch_share if batch_share > 0 else None
    else:
        panel["clq14_top_decile_share"] = None
        panel["clq14_batch_share"] = None
        panel["clq14_enrichment"] = None

    if labels is not None:  # capture: did the queue surface the true fails?
        df = scores.merge(labels, on=schema.KEY_COL, how="left")
        y = df[schema.TARGET_COL].eq(1).to_numpy()  # raw -1/1 -> 0/1
        n_fails = int(y.sum())
        panel["n_fails"] = n_fails
        panel["base_rate"] = float(y.mean())
        top = df.loc[df["decile"] == 1, schema.TARGET_COL].eq(1)
        fails_top = int(top.sum())
        panel["top_decile_fails"] = fails_top
        panel["top_decile_capture"] = fails_top / n_fails if n_fails else None
        panel["top_decile_lift"] = (
            (fails_top / len(top)) / panel["base_rate"] if n_fails and panel["base_rate"] else None
        )
        for q in flag_rates:
            r = recall_at_flagrate(y, p.to_numpy(), q=q)
            panel[f"recall_at_flag_{q:.2f}"] = float(r[0]) if n_fails else None
    else:
        for k in (
            "n_fails",
            "base_rate",
            "top_decile_fails",
            "top_decile_capture",
            "top_decile_lift",
        ):
            panel[k] = None
        for q in flag_rates:
            panel[f"recall_at_flag_{q:.2f}"] = None
    return panel


def add_deltas(panel: dict, ref_panel: dict, ref_id: str) -> None:
    panel["reference"] = ref_id
    for k in ("p_median", "p_mean", "p90", "top_decile_capture"):
        a, b = panel.get(k), ref_panel.get(k)
        panel[f"{k}_delta_vs_ref"] = a - b if (a is not None and b is not None) else None


def panel_for_run(engine, run_dir: Path) -> dict:
    scores = pd.read_parquet(run_dir / "scores.parquet")
    cfg = json.loads((run_dir / "config.json").read_text())["config"]
    labels = fetch_labels(engine, scores[schema.KEY_COL].to_numpy())
    flags = fetch_flags(engine, cfg)
    panel = compute_panel(scores, labels, flags)
    panel["score_run"] = run_dir.name
    panel["label"] = cfg.get("label", "")
    return panel


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "scorecard.log")
    ap = argparse.ArgumentParser(description="Evaluate a scored batch against its true labels.")
    ap.add_argument("--score-run", required=True, help="score run id or 'latest'")
    ap.add_argument(
        "--label", default=None, help="with --score-run latest: pick latest of this label"
    )
    ap.add_argument("--reference", default=None, help="reference score run id for deltas")
    ap.add_argument("--reference-label", default=None, help="reference label (implies latest)")
    args = ap.parse_args()

    engine = get_engine()
    run_dir = resolve_score_run(args.score_run, args.label)
    logger.info("[scorecard] evaluating %s", run_dir.name)
    panel = panel_for_run(engine, run_dir)

    ref_label = args.reference_label
    ref_run = args.reference or ("latest" if ref_label else None)
    if ref_run:
        ref_dir = resolve_score_run(ref_run, ref_label)
        add_deltas(panel, panel_for_run(engine, ref_dir), ref_dir.name)

    (run_dir / "scorecard.json").write_text(json.dumps(panel, indent=2))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # local — matches tracking.make_run
    capture = panel.get("top_decile_capture")
    lift = panel.get("top_decile_lift")
    note = (
        f"label={panel['label']}; n={panel['n_scored']}; capture={capture:.2f} lift={lift:.1f}"
        if capture is not None and lift is not None
        else f"label={panel['label']}; n={panel['n_scored']}; unlabeled"
    )
    append_index(
        {
            "run_id": f"{ts}_scorecard__{panel['label']}",
            "git_sha": git_sha(),
            "note": note,
            "type": "scorecard",
            "parent_run": run_dir.name,
        }
    )
    logger.info("[scorecard] %s", note)
    logger.info("[scorecard] done -> %s", run_dir / "scorecard.json")


if __name__ == "__main__":
    main()
