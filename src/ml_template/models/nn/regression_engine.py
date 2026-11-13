"""MSE training with continuous outputs; binary training engine is unchanged."""

import logging

import numpy as np
import torch
from torch import nn

from ml_template.models.nn.backbones import build_backbone
from ml_template.models.nn.data import batch_indices
from ml_template.models.nn.engine import TrainingResult, _input_tensors, _logits, execution_context
from ml_template.models.nn.regression import RegressionStop, metric_value

logger = logging.getLogger(__name__)


def train(
    inputs,
    labels,
    params,
    seed,
    n_epochs,
    *,
    validation=None,
    early_stopping=False,
    curve_metric=None,
):
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, (int, np.integer)) or n_epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if early_stopping and validation is None:
        raise ValueError("early stopping requires validation")
    if curve_metric is not None and curve_metric not in {"rmse", "mae"}:
        raise ValueError("regression curves require rmse or mae")
    history, selected = [], int(n_epochs)
    model, optimizer = None, None
    with execution_context(params, seed, params.device) as device:
        try:
            model = build_backbone(inputs.numeric.shape[1], inputs.cardinalities, params).to(device)
            tensors = _input_tensors(inputs)
            target_tensor = torch.from_numpy(labels)
            validation_tensors = _input_tensors(validation[0]) if validation is not None else None
            criterion = nn.MSELoss()
            optimizer_class = torch.optim.Adam if params.optimizer == "adam" else torch.optim.AdamW
            optimizer = optimizer_class(
                model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
            )
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.max_epochs)
                if params.scheduler == "cosine"
                else None
            )
            stopper = RegressionStop(params.patience, params.min_delta)
            rng, best_state = np.random.default_rng(seed), None
            for epoch in range(1, int(n_epochs) + 1):
                model.train()
                for idx in batch_indices(len(labels), params.batch_size, rng):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(tensors[0][idx].to(device), tensors[1][idx].to(device))
                    loss = criterion(prediction, target_tensor[idx].to(device))
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("nonfinite NN regression loss")
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("nonfinite NN regression gradients")
                    if params.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), params.gradient_clip_norm, error_if_nonfinite=True
                        )
                    optimizer.step()
                if any(not bool(torch.isfinite(p).all().item()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite NN regression weights")
                model.refresh_unknown_embeddings()
                train_prediction = _logits(model, tensors, device, params.prediction_batch_size)
                for metric in dict.fromkeys(["mse", params.monitor, curve_metric]):
                    if metric is not None:
                        history.append(
                            {
                                "round": epoch,
                                "dataset": "train",
                                "metric": metric,
                                "value": metric_value(metric, labels, train_prediction),
                            }
                        )
                if validation is not None:
                    prediction = _logits(
                        model, validation_tensors, device, params.prediction_batch_size
                    )
                    score = metric_value(params.monitor, validation[1], prediction)
                    for metric in dict.fromkeys([params.monitor, curve_metric]):
                        if metric is not None:
                            history.append(
                                {
                                    "round": epoch,
                                    "dataset": "val",
                                    "metric": metric,
                                    "value": metric_value(metric, validation[1], prediction),
                                }
                            )
                    logger.info("NN regression epoch %d: val_%s=%.6f", epoch, params.monitor, score)
                    if early_stopping:
                        if stopper.update(score, epoch):
                            best_state = {
                                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            }
                        if stopper.stopped:
                            break
                if scheduler is not None and epoch < params.max_epochs:
                    scheduler.step()
            if early_stopping:
                if best_state is None:
                    raise RuntimeError("no finite NN regression checkpoint")
                model.load_state_dict(best_state)
                selected = stopper.best_epoch
            model.eval()
        finally:
            optimizer = None
            if model is not None:
                model.cpu()
    return TrainingResult(
        model,
        selected,
        history,
        {
            "torch_version": str(torch.__version__),
            "training_device": params.device,
            "deterministic": params.deterministic,
            "precision": "float32",
            "seed": int(seed),
            "task": "regression",
            "loss": "mse",
        },
    )
