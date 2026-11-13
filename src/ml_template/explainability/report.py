"""Static presentation from explanation artifacts, never model execution."""

from html import escape
from pathlib import Path


def write_report(directory, summary, ranking):
    names = ("explanation_id", "release_id", "scoring_batch_id", "n_rows", "n_features")
    lines = ["# Model explanation", ""]
    lines.extend(f"- {name}: {summary[name]}" for name in names)
    lines += [
        "",
        "## Interpretation",
        "",
        "Contributions explain base-model log-odds, not calibrated probability or policy.",
        "They describe model behaviour, not causes or evidence of predictive accuracy.",
        "The population is the complete original scoring batch; no labels were loaded.",
        "Correlated and engineered inputs may share attribution.",
        "",
        "## Largest mean absolute contributions",
        "",
        "Feature | Mean absolute contribution (log-odds)",
        "--- | ---",
    ]
    for row in ranking.head(15).itertuples():
        name = str(row.feature).replace("|", r"\|").replace("\n", " ")
        lines.append(f"{name} | {row.mean_abs_contribution:.6g}")
    text = "\n".join(lines) + "\n"
    directory = Path(directory)
    (directory / "report.md").write_text(text)
    (directory / "report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Model explanation</title><body><pre>" + escape(text) + "</pre></body></html>"
    )
