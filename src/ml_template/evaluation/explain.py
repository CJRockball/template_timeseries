from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBClassifier


def save_shap_plots(
    model: XGBClassifier,
    X: pd.DataFrame,
    output_dir: Path,
    max_display: int = 25,
) -> pd.DataFrame:
    """Compute TreeSHAP, save core plots, and return ranked feature summary."""

    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    summary = (
        pd.Series(
            abs(shap_values).mean(axis=0),
            index=X.columns,
            name="mean_abs_shap",
        )
        .sort_values(ascending=False)
        .to_frame()
    )

    summary.to_parquet(output_dir / "shap_feature_summary.parquet")
    summary.to_csv(output_dir / "shap_feature_summary.csv")

    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    return summary
