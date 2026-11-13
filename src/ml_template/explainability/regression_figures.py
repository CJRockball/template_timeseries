"""Pure sampled signed-contribution scatter distributions in target units."""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def distribution_figure(view, *, top_n=10, max_points=2000):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (top_n, max_points)):
        raise ValueError("positive figure limits required")
    order = np.argsort(view.rows.index.to_numpy(), kind="stable")
    positions = order[np.linspace(0, len(order) - 1, min(max_points, len(order)), dtype=int)]
    figure = go.Figure()
    top = view.importance.head(top_n).feature.tolist()[::-1]
    for i, name in enumerate(top):
        values = view.features[name].iloc[positions]
        numeric = pd.api.types.is_numeric_dtype(values.dtype)
        colors = None
        if numeric and name not in view.summary.get("varying_feature_values", []):
            observed = values.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(observed)
            if finite.any():
                low, high = np.min(observed[finite]), np.max(observed[finite])
                colors = np.full(len(values), 0.5)
                colors[finite] = (observed[finite] - low) / (high - low) if high > low else 0.5
        jitter = ((np.arange(len(positions)) * 0.61803398875) % 1 - 0.5) * 0.55
        marker = {"size": 5, "opacity": 0.65}
        if colors is not None:
            marker.update(color=colors, colorscale="RdBu_r", cmin=0, cmax=1)
        else:
            marker.update(color="gray")
        figure.add_trace(
            go.Scattergl(
                x=view.contributions[name].iloc[positions],
                y=i + jitter,
                mode="markers",
                name=escape(name),
                showlegend=False,
                marker=marker,
                text=[escape(str(key)) for key in view.rows.index[positions]],
                hovertemplate="Entity: %{text}<br>Contribution: %{x}<extra></extra>",
            )
        )
    figure.add_vline(x=0, line_dash="dot", line_color="gray")
    figure.update_layout(
        template="plotly_white",
        height=max(400, 45 * len(top)),
        title=f"Signed SHAP distributions — {len(positions)} of {len(view.rows)} rows (jittered)",
        xaxis_title="Contribution (" + escape(view.summary["target_unit"]) + ")",
        yaxis={"tickvals": list(range(len(top))), "ticktext": [escape(n) for n in top]},
    )
    return figure
