"Fresh per-origin stopping/refit with the existing regression adapter."

import json

import numpy as np
import pandas as pd

from .learned_artifacts import Publication, rows, verify, write_line
from .temporal_samples import number


def frame(records, features, cap, labels=True):
    selected = []
    targets = []
    for row in records:
        if len(selected) >= cap:
            raise ValueError("Dense matrix cap exceeded")
        if row["x"] is None or len(row["x"]) != len(features):
            raise ValueError("Invalid materialised feature vector")
        selected.append([number(v) for v in row["x"]])
        if labels:
            targets.append(number(row["y"]))
    if not selected:
        raise ValueError("Empty materialised matrix")
    return (pd.DataFrame(selected, columns=features, dtype=float), pd.Series(targets, dtype=float))


def train(sample_path, parameters, seed, producer, factory, reload_model, output_root, database):
    """factory/reload_model are private seams for native adapters or explicit test doubles."""
    manifest = verify(sample_path, "samples")
    context = manifest["identity"]["inputs"]
    if context["producer"] != producer:
        raise ValueError("Sample producer changed; rematerialise instead of bypassing identity")
    features = context["sample_options"]["features"]
    cap = context["sample_options"]["max_rows_per_partition"]
    publication = Publication(output_root)
    models = []
    try:
        (publication.stage / "models").mkdir()
        with (publication.stage / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
            for ordinal, window in enumerate(context["windows"]):
                origin = window["origin_date"]
                if window["target_date"] >= context["holdout"]["start"]:
                    raise ValueError("Holdout prediction prohibited")

                def partition(name, selected_origin=origin):
                    return (
                        row
                        for row in rows(sample_path, "samples.jsonl")
                        if row["origin"] == selected_origin and row["partition"] == name
                    )

                Xfit, yfit = frame(partition("fit"), features, cap)
                Xstop, ystop = frame(partition("stopping"), features, cap)
                stopper = factory()
                rounds = stopper.fit_with_validation(Xfit, yfit, Xstop, ystop)
                if type(rounds) is not int or rounds < 1 or rounds > parameters["n_estimators"]:
                    raise ValueError("Adapter returned invalid selected round count")
                history = getattr(stopper, "early_stopping_history", None)
                if history is not None:
                    history.to_csv(
                        publication.stage / f"models/{ordinal:04d}_stopping.csv", index=False
                    )
                del Xfit, yfit, Xstop, ystop, stopper
                Xrefit, yrefit = frame(partition("refit"), features, cap)
                model = factory()
                model.refit(Xrefit, yrefit, rounds)
                refit_rows = len(Xrefit)
                del Xrefit, yrefit
                requests = [
                    row
                    for row in rows(sample_path, "requests.jsonl")
                    if row["request"]["origin_date"] == origin
                ]
                if len(requests) > cap:
                    raise ValueError("Forecast matrix cap exceeded")
                supported = [row for row in requests if row["x"] is not None]
                Xpred, _ = frame(supported, features, cap, False)
                before = np.asarray(model.predict(Xpred), dtype=float)
                model_file = f"models/{ordinal:04d}.ubj"
                model.save(publication.stage / model_file)
                loaded = reload_model(publication.stage / model_file)
                after = np.asarray(loaded.predict(Xpred), dtype=float)
                if (
                    before.shape != (len(supported),)
                    or after.shape != before.shape
                    or (not np.isfinite(before).all())
                    or (not np.array_equal(before, after))
                ):
                    raise ValueError("Native save/load prediction parity failed")
                values = iter(before.tolist())
                for row in requests:
                    request = row["request"]
                    ok = row["x"] is not None
                    write_line(
                        predictions,
                        {
                            **request,
                            "method": "xgb",
                            "status": "ok" if ok else "unsupported",
                            "y_pred": number(next(values)) if ok else None,
                            "reason": None if ok else "incomplete_selected_features",
                        },
                    )
                models.append(
                    {
                        "origin": origin,
                        "model_file": model_file,
                        "selected_rounds": rounds,
                        "refit_rows": refit_rows,
                        "requests": len(requests),
                        "supported": len(supported),
                        "feature_order": features,
                        **(
                            {
                                "requested_training_device": "cuda",
                                "effective_training_device": model.effective_training_device,
                            }
                            if parameters.get("device") == "cuda"
                            else {}
                        ),
                    }
                )
        (publication.stage / "report.json").write_text(
            json.dumps(
                {
                    "models": models,
                    "round_selection": "chronological_inner_stopping_then_fresh_refit",
                    "holdout_scored": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        binding = {
            "context": context,
            "samples_id": manifest["artifact_id"],
            "samples_manifest_sha256": manifest["manifest_sha256"],
            "parameters": parameters,
            "seed": seed,
            "producer": producer,
        }
        return publication.publish("models", binding, database)
    finally:
        publication.close()
