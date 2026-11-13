"""Shared read-only Dash panels for exploratory OOF threshold recommendations."""

from __future__ import annotations

import pandas as pd
from dash import dcc, html

from ml_template.dashboard import threshold_figs as tf
from ml_template.dashboard.inventory_common import grid
from ml_template.evaluation.threshold import (
    budget_boundaries,
    fold_recommendations,
    recommendations,
    threshold_profile,
)

DISCLAIMER = (
    "Estimated from these same OOF rows: maximised scores are exploratory and selection-biased, "
    "not independently validated. This does not calibrate scores, freeze a policy, change the slider "
    "or access holdout outcomes. Near-best envelopes use an absolute F-score tolerance of 0.01; "
    "they may contain gaps, are not confidence intervals, and are not necessarily stable deployment ranges."
)


def render_threshold_panel(predictions, current_threshold=0.5, budget=0.05, *, metadata=None):
    if not predictions:
        return html.P(
            "Threshold estimates unavailable: complete validated development OOF predictions are required.",
            role="status",
        )
    metadata = metadata or {}
    profiles, recs, folds, budgets = {}, [], [], []
    for run_id, frame in predictions.items():
        if frame.empty:
            continue
        # Prediction loaders have already checked development membership and aligned keys/folds.
        profile = threshold_profile(frame["y_true"], frame["p_oof"])
        profiles[run_id] = profile
        recs.append(recommendations(profile).assign(run_id=run_id, **metadata.get(run_id, {})))
        folded = fold_recommendations(frame["y_true"], frame["p_oof"], frame["fold"])
        folds.append(folded.assign(run_id=run_id, **metadata.get(run_id, {})))
        budgets.append(
            budget_boundaries(frame["y_true"], frame["p_oof"], rates=(0.05, 0.10, budget)).assign(
                run_id=run_id, **metadata.get(run_id, {})
            )
        )
    if not profiles:
        return html.P("No OOF rows available for threshold estimates", role="status")
    recommendations_table = pd.concat(recs, ignore_index=True)
    display = recommendations_table.copy()
    if "threshold" in display:
        display["cutoff_display"] = display["threshold"].map(
            lambda value: (
                "undefined"
                if pd.isna(value)
                else (">1 (flag none)" if value > 1 else f"{value:.10g}")
            )
        )
    tabs = [
        dcc.Tab(
            label=objective.upper(),
            children=[
                dcc.Graph(figure=tf.objective_overlay(profiles, objective, current_threshold))
            ],
        )
        for objective in ("f1", "f2")
    ]
    for run_id, profile in profiles.items():
        fold_table = pd.concat(folds, ignore_index=True)
        fold_table = fold_table[fold_table["run_id"] == run_id]
        tabs.append(
            dcc.Tab(
                label=f"Workload / folds: {run_id}",
                children=[
                    dcc.Graph(figure=tf.operating_curves(profile, current_threshold)),
                    dcc.Graph(figure=tf.fold_optima(fold_table)),
                    grid(fold_table),
                ],
            )
        )
    return html.Div(
        [
            html.P(DISCLAIMER),
            html.P(
                "The cutoff applies only to the named score variant. A calibrated-cross-fit "
                "recommendation is not an exact transfer guarantee for one final calibrator. "
                "No raw cutoff is numerically reused on another score scale."
            ),
            html.P(
                "F1 balances precision/recall; F2 weights recall more heavily. Candidates use exact observed "
                "scores, not the slider's 0.01 grid. Tied maxima choose the highest candidate cutoff. "
                "Fold optima are separately selected descriptive sensitivities, not held-out policy scores."
            ),
            grid(display, 300),
            dcc.Tabs(children=tabs),
            html.H4("Equal-budget score boundaries (5%, 10% and current budget)"),
            html.P(
                "A boundary score is an exact scalar cutoff only when threshold_exact is true. "
                "If a boundary tie crosses the budget, top-k uses ascending entity-key order, "
                "while p >= boundary would flag all tied rows. A scalar cutoff cannot reproduce that split."
            ),
            grid(pd.concat(budgets, ignore_index=True), 320),
        ]
    )


def render_single(engine, run_id, threshold):
    from ml_template.dashboard import run_data as rd

    if not run_id:
        return html.P("Choose a run for threshold estimates", role="status")
    try:
        header = rd.run_header(engine, run_id)
        frame = rd.load_oof_with_labels(engine, header, rd.run_dir(header))
        return render_threshold_panel({run_id: frame} if frame is not None else {}, threshold)
    except (ValueError, KeyError, OSError) as exc:
        return html.P(f"Threshold estimation blocked: {exc}", role="status")


# calibrated-policy update v1
