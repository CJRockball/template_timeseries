"""AI4I Predictive Maintenance -- EDA script.

Converted from notebooks/eda_small.ipynb. Reads Bronze data via
extract_bronze(), runs each EDA section as a pure function that returns a
JSON-able result, writes plots to files instead of plt.show(), and dumps
one machine-readable report (JSON) plus one human-readable report (Markdown)
per run. All findings feed the Silver contract in configs/ai4i_binary.yaml.

Usage:
    python scripts/eda_script.py --dataset-version-id ds_v1 --out artifacts/eda/ds_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never call plt.show() in a script
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif

from ml_template.data.extract import extract_bronze
from ml_template.db import data_run_registry
from ml_template.db.connection import get_engine
from ml_template.tracking.pipeline_logging import setup_logging

logger = logging.getLogger("ml_template.scripts.eda")

BI_COLORS = ["#008BFB", "#FF0051"]
CMAP_DIV = LinearSegmentedColormap.from_list("shap_div", ["#008BFB", "#ffffff", "#FF0051"], N=512)


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #
@dataclass
class EdaReport:
    dataset_version_id: str
    row_count: int = 0
    col_count: int = 0
    duplicate_rows: int = 0
    target_outcome_summary: dict[str, Any] = field(default_factory=dict)
    label_consistency: dict[str, Any] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    numeric_summary: dict[str, Any] = field(default_factory=dict)
    categorical_summary: dict[str, Any] = field(default_factory=dict)
    correlations: dict[str, Any] = field(default_factory=dict)
    mutual_information: dict[str, float] = field(default_factory=dict)
    outliers: dict[str, Any] = field(default_factory=dict)
    outlier_failure_rates: dict[str, float] = field(default_factory=dict)
    interaction_candidates: list[dict[str, Any]] = field(default_factory=list)
    proposed_silver_contract: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))


# --------------------------------------------------------------------------- #
# Section 1 -- setup / load / canonicalize
# --------------------------------------------------------------------------- #
def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_and_canonicalize(
    dataset_version_id: str, data_config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    engine = get_engine()
    df = extract_bronze(dataset_version_id=dataset_version_id, engine=engine)

    canonical_names = data_config["silver_eda_contract"]["canonical_names"]
    canonical_dtypes = data_config["silver_eda_contract"]["canonical_dtypes"]
    df = df.rename(columns=canonical_names).astype(canonical_dtypes)

    roles = data_config["bronze_source_contract"]["roles"]
    role_map = {
        role: [canonical_names.get(col, col) for col in cols] for role, cols in roles.items()
    }
    logger.info(
        "Loaded %d rows, %d columns for dataset_version_id=%s",
        len(df),
        df.shape[1],
        dataset_version_id,
    )
    return df, role_map


def check_duplicates(df: pd.DataFrame) -> int:
    n = int(df.duplicated().sum())
    if n:
        logger.warning("Found %d duplicate rows", n)
    return n


# --------------------------------------------------------------------------- #
# Section 2 -- target and outcomes
# --------------------------------------------------------------------------- #
def summarize_targets(df: pd.DataFrame, all_targets: list[str]) -> dict[str, Any]:
    vc_list = [df[name].value_counts().reindex([0, 1], fill_value=0) for name in all_targets]
    vc_df = pd.concat(vc_list, axis=1, keys=all_targets).T
    n = len(df)
    vc_df["proportion_%"] = round(vc_df.get(1, 0) / n * 100, 2)
    vc_df["ratio"] = round(vc_df.get(0, 0) / vc_df.get(1, 1), 2)
    return vc_df.to_dict(orient="index")


def check_label_consistency(df: pd.DataFrame, target: str, outcome: list[str]) -> dict[str, Any]:
    n_fmodes = df[outcome].sum(axis=1)
    consistency = {
        "failure_mode_positive_no_target": int(((n_fmodes > 0) & (df[target] == 0)).sum()),
        "target_positive_no_failure_mode": int(((n_fmodes == 0) & (df[target] == 1)).sum()),
        "target_positive_with_failure_mode": int(((n_fmodes > 0) & (df[target] == 1)).sum()),
        "compound_failure_rows": int((n_fmodes > 1).sum()),
    }
    return consistency


def plot_target_grid(df: pd.DataFrame, all_targets: list[str], out_dir: Path) -> Path:
    import math

    n_cols = 2
    n_rows = math.ceil(len(all_targets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows), squeeze=False)

    for idx, col in enumerate(all_targets):
        ax = axes[idx // n_cols, idx % n_cols]
        vc = df[col].astype(str).value_counts().reindex(["0", "1"], fill_value=0)
        pct = (vc / vc.sum()) * 100
        bars = ax.bar(vc.index, vc.values, color=BI_COLORS, edgecolor="white", linewidth=0.8)
        labels = [f"{c:,}\n({p:.1f}%)" for c, p in zip(vc.values, pct.values, strict=True)]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=9, fontweight="bold")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vc.values) * 1.18)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for idx in range(len(all_targets), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])

    plt.tight_layout()
    out_path = out_dir / "target_outcome_distribution.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Section 3 -- missing values / sentinel checks
# --------------------------------------------------------------------------- #
SENTINEL_CANDIDATES = [-1, -99, -999, -9999, 999, 9999, 99999]


def check_missing_and_sentinels(df: pd.DataFrame, num_feature: list[str]) -> dict[str, Any]:
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    sentinel_hits: dict[str, list[float]] = {}
    for col in num_feature:
        found = [v for v in SENTINEL_CANDIDATES if (df[col] == v).any()]
        if found:
            sentinel_hits[col] = found

    negative_on_positive_domain = {
        col: int((df[col] < 0).sum()) for col in num_feature if (df[col] < 0).any()
    }

    return {
        "missing_counts": missing,
        "sentinel_hits": sentinel_hits,
        "negative_on_positive_domain": negative_on_positive_domain,
    }


# --------------------------------------------------------------------------- #
# Section 4 -- feature distributions
# --------------------------------------------------------------------------- #
def summarize_numeric(df: pd.DataFrame, num_feature: list[str]) -> dict[str, Any]:
    desc = df[num_feature].describe().to_dict()
    coef_var = (df[num_feature].std() / df[num_feature].mean()).round(6).to_dict()
    skew = df[num_feature].skew().round(4).to_dict()
    kurtosis = df[num_feature].kurtosis().round(4).to_dict()
    return {
        "describe": desc,
        "coef_var": coef_var,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def summarize_categorical(df: pd.DataFrame, cat_feature: list[str]) -> dict[str, Any]:
    return {col: df[col].value_counts().to_dict() for col in cat_feature}


def plot_violin_by_target(
    df: pd.DataFrame, num_feature: list[str], target: str, out_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, len(num_feature), figsize=(4 * len(num_feature), 5), sharey=False)
    for ax, col in zip(np.atleast_1d(axes), num_feature, strict=True):
        sns.violinplot(data=df, x=target, y=col, hue=target, ax=ax, legend=False, palette=BI_COLORS)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
    plt.tight_layout()
    out_path = out_dir / "numeric_by_target_violin.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Section 5 -- correlations (Spearman, Cramer's V, Mutual Information)
# --------------------------------------------------------------------------- #
def spearman_matrix(df: pd.DataFrame, num_feature: list[str]) -> pd.DataFrame:
    return df[num_feature].corr(method="spearman")


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(confusion, correction=False)
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2corr / denom))


def cramers_v_matrix(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    n = len(cat_cols)
    v_matrix = pd.DataFrame(np.eye(n), index=cat_cols, columns=cat_cols)
    for i, j in combinations(range(n), 2):
        v = _cramers_v(df[cat_cols[i]], df[cat_cols[j]])
        v_matrix.iloc[i, j] = v
        v_matrix.iloc[j, i] = v
    return v_matrix


def mutual_information_scores(
    df: pd.DataFrame, feature: list[str], cat_feature: list[str], target: str
) -> dict[str, float]:
    x = df[feature].copy()
    for col in cat_feature:
        x[col] = x[col].astype("category").cat.codes
    discrete_mask = [col in cat_feature for col in feature]
    mi = mutual_info_classif(x, df[target], discrete_features=discrete_mask, random_state=0)
    return dict(sorted(zip(feature, mi.tolist(), strict=True), key=lambda kv: -kv[1]))


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path) -> Path:
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    fig = plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=CMAP_DIV,
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=13)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Section 6 -- outliers (IQR + IsolationForest)
# --------------------------------------------------------------------------- #
def detect_iqr_outliers(df: pd.DataFrame, num_feature: list[str]) -> dict[str, Any]:
    outlier_info: dict[str, Any] = {}
    for col in num_feature:
        s = df[col].astype(float)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        outlier_info[col] = {
            "low_thresh": round(float(lo), 2),
            "high_thresh": round(float(hi), 2),
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 3),
            "outlier_rows": df.index[mask].tolist(),
        }
    return outlier_info


def outlier_failure_rates(
    df: pd.DataFrame, outlier_info: dict[str, Any], target: str
) -> dict[str, float]:
    rates = {}
    for col, stats in outlier_info.items():
        rows = stats["outlier_rows"]
        rates[col] = round(float(df.loc[rows, target].mean() * 100), 2) if rows else float("nan")
    return rates


def isolation_forest_scores(
    df: pd.DataFrame, num_feature: list[str], contamination: float = 0.02
) -> pd.Series:
    iso = IsolationForest(contamination=contamination, random_state=0)
    iso.fit(df[num_feature])
    scores = iso.decision_function(df[num_feature])
    return pd.Series(scores, index=df.index, name="iso_score")


# --------------------------------------------------------------------------- #
# Section 7 -- interactions (quantile-binned joint failure rate)
# --------------------------------------------------------------------------- #
def make_interaction_bins(
    data: pd.DataFrame, feature_1: str, feature_2: str, n_bins: int = 6, precision: int = 2
) -> tuple[list[float], list[str], list[float], list[str]]:
    def _bins_and_labels(feature: str) -> tuple[list[float], list[str]]:
        values = pd.to_numeric(data[feature], errors="coerce").dropna()
        _, edges = pd.qcut(values, q=n_bins, labels=False, retbins=True, duplicates="drop")
        bins = edges.tolist()
        labels = [
            f"{left:.{precision}f}\u2013{right:.{precision}f}"
            for left, right in zip(bins[:-1], bins[1:], strict=True)
        ]
        return bins, labels

    b1, l1 = _bins_and_labels(feature_1)
    b2, l2 = _bins_and_labels(feature_2)
    return b1, l1, b2, l2


def interaction_failure_table(
    df: pd.DataFrame, feature_1: str, feature_2: str, target: str, n_bins: int = 6
) -> dict[str, Any]:
    b1, l1, b2, l2 = make_interaction_bins(df, feature_1, feature_2, n_bins)
    binned = df.copy()
    binned[f"{feature_1}_bin"] = pd.cut(df[feature_1], bins=b1, labels=l1, include_lowest=True)
    binned[f"{feature_2}_bin"] = pd.cut(df[feature_2], bins=b2, labels=l2, include_lowest=True)

    failure_rate = pd.crosstab(
        binned[f"{feature_1}_bin"],
        binned[f"{feature_2}_bin"],
        values=binned[target],
        aggfunc="mean",
    )
    cell_count = pd.crosstab(binned[f"{feature_1}_bin"], binned[f"{feature_2}_bin"])
    return {
        "pair": [feature_1, feature_2],
        "failure_rate": failure_rate.round(4).to_dict(),
        "cell_count": cell_count.to_dict(),
        "min_cell_count": int(cell_count.min().min()),
    }


def screen_interaction_candidates(
    mi_scores: dict[str, float], spearman: pd.DataFrame, top_n: int = 5, corr_ceiling: float = 0.8
) -> list[tuple[str, str]]:
    ranked = [f for f, _ in sorted(mi_scores.items(), key=lambda kv: -kv[1])][:top_n]
    candidates = []
    for f1, f2 in combinations(ranked, 2):
        if (
            f1 in spearman.columns
            and f2 in spearman.columns
            and abs(spearman.loc[f1, f2]) < corr_ceiling
        ):
            candidates.append((f1, f2))
    return candidates


# --------------------------------------------------------------------------- #
# Section 8 -- proposed Silver contract (data-level decisions only)
# --------------------------------------------------------------------------- #
def propose_silver_contract(report: EdaReport, data_config: dict[str, Any]) -> dict[str, Any]:
    """Emit a Silver-ready contract fragment from EDA findings.

    Only deterministic, data-level decisions are included here (naming,
    dtypes, roles, row-count expectation, nullability policy). Model-fitted
    transforms (imputation, scaling, encoding, interaction features) are
    deliberately excluded -- they belong in the model pipeline.
    """
    return {
        "canonical_names": data_config["silver_eda_contract"]["canonical_names"],
        "canonical_dtypes": data_config["silver_eda_contract"]["canonical_dtypes"],
        "roles": data_config["bronze_source_contract"]["roles"],
        "expected_row_count": data_config["bronze_source_contract"]["rows"],
        "reported_row_count": report.row_count,
        "row_count_severity": "error",
        "nullability_policy": "no_structural_nulls_expected",
        "sentinel_values_found": report.missing_values.get("sentinel_hits", {}),
        "duplicate_rows_found": report.duplicate_rows,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(dataset_version_id: str, config_path: Path, out_dir: Path) -> EdaReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_config = load_config(config_path)

    df, roles = load_and_canonicalize(dataset_version_id, data_config)
    target = roles["target"][0]
    outcome = roles["outcome"]
    feature = roles["feature"]
    cat_feature = [c for c in feature if str(df[c].dtype) in ("object", "category", "string")]
    num_feature = [c for c in feature if c not in cat_feature]
    all_targets = [target] + outcome

    report = EdaReport(
        dataset_version_id=dataset_version_id, row_count=len(df), col_count=df.shape[1]
    )

    # Section 1
    report.duplicate_rows = check_duplicates(df)

    # Section 2
    report.target_outcome_summary = summarize_targets(df, all_targets)
    report.label_consistency = check_label_consistency(df, target, outcome)
    plot_target_grid(df, all_targets, out_dir)

    # Section 3
    report.missing_values = check_missing_and_sentinels(df, num_feature)

    # Section 4
    report.numeric_summary = summarize_numeric(df, num_feature)
    report.categorical_summary = summarize_categorical(df, cat_feature)
    plot_violin_by_target(df, num_feature, target, out_dir)

    # Section 5
    spearman = spearman_matrix(df, num_feature)
    plot_heatmap(spearman, "Spearman Correlation - Numeric Features", out_dir / "spearman.png")
    report.correlations["spearman"] = spearman.round(4).to_dict()

    if len(cat_feature) > 1:
        cv = cramers_v_matrix(df, cat_feature)
        plot_heatmap(cv, "Cramer's V - Categorical Features", out_dir / "cramers_v.png")
        report.correlations["cramers_v"] = cv.round(4).to_dict()

    report.mutual_information = mutual_information_scores(df, feature, cat_feature, target)

    # Section 6
    outlier_info = detect_iqr_outliers(df, num_feature)
    report.outliers = {
        col: {k: v for k, v in stats.items() if k != "outlier_rows"}
        for col, stats in outlier_info.items()
    }
    report.outlier_failure_rates = outlier_failure_rates(df, outlier_info, target)
    df["_iso_score"] = isolation_forest_scores(df, num_feature)

    # Section 7
    candidates = screen_interaction_candidates(report.mutual_information, spearman)
    for f1, f2 in candidates:
        report.interaction_candidates.append(interaction_failure_table(df, f1, f2, target))

    # Section 8 -- Silver proposal
    report.proposed_silver_contract = propose_silver_contract(report, data_config)

    report_path = out_dir / f"eda_report_{dataset_version_id}.json"
    report.to_json(report_path)
    logger.info("EDA report written to %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI4I EDA against a pinned Bronze version.")
    parser.add_argument("--dataset-version-id", required=True)
    parser.add_argument("--config", default="configs/ai4i_binary.yaml", type=Path)
    parser.add_argument("--out", default="artifacts/eda", type=Path)
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting EDA run for dataset_version_id=%s", args.dataset_version_id)

    out_dir = args.out / args.dataset_version_id

    engine = get_engine()
    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    run_id = data_run_registry.start_eda_run(
        engine, args.dataset_version_id, config_sha256=config_sha256
    )
    report_path = out_dir / f"eda_report_{args.dataset_version_id}.json"
    try:
        run_eda(args.dataset_version_id, args.config, out_dir)
        saved = json.loads(report_path.read_text())
    except Exception as exc:
        data_run_registry.finish_eda_run(
            engine, run_id, succeeded=False, error_summary=str(exc)[:500]
        )
        raise
    data_run_registry.finish_eda_run(
        engine,
        run_id,
        succeeded=True,
        row_count=saved.get("row_count"),
        report_path=str(report_path),
    )
    logger.info("EDA completed: %s", report_path)


if __name__ == "__main__":
    main()
