"""PyTorch trainer: stable loss, finite gradients and CPU-offloaded models."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ml_template.models.nn.backbones import build_backbone
from ml_template.models.nn.data import batch_indices, resolve_pos_weight
from ml_template.models.nn.metrics import EarlyStop, metric_value, probabilities

logger = logging.getLogger(__name__)


def requested_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name not in ("cuda", "cuda:0"):
        raise ValueError("device must be cpu, cuda or cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; choose device: cpu")
    return torch.device("cuda:0")


@contextmanager
def execution_context(params, seed, device_name):
    device = requested_device(device_name)
    if (
        device.type == "cuda"
        and params.deterministic
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8")
    ):
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching deterministic CUDA"
        )
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_state = None
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(params.cpu_threads)
        torch.use_deterministic_algorithms(params.deterministic)
        if device.type == "cuda":
            cudnn_state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
            torch.backends.cudnn.deterministic = params.deterministic
            torch.backends.cudnn.benchmark = False
        scope = torch.cuda.device(0) if device.type == "cuda" else nullcontext()
        with scope, torch.random.fork_rng(devices=[] if device.type == "cpu" else [0]):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            yield device
    finally:
        if cudnn_state is not None:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_state
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)
        torch.set_num_threads(previous_threads)
        torch.set_default_dtype(previous_dtype)


def _input_tensors(inputs):
    return (
        torch.tensor(inputs.numeric, dtype=torch.float32, device="cpu"),
        torch.tensor(inputs.categorical, dtype=torch.int64, device="cpu"),
    )


@torch.inference_mode()
def _logits(model, tensors, device, batch_size):
    model.eval()
    numeric, categorical = tensors
    parts = []
    for idx in batch_indices(len(numeric), batch_size):
        out = model(numeric[idx].to(device), categorical[idx].to(device))
        parts.append(out.cpu().numpy().astype(np.float64))
    result = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite NN logits")
    return result


@dataclass
class TrainingResult:
    model: object
    best_epoch: int
    history: list[dict]
    execution: dict


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
    if n_epochs < 1 or (early_stopping and validation is None):
        raise ValueError("invalid epoch/validation request")
    model, optimizer = None, None
    history = []
    selected = int(n_epochs)
    with execution_context(params, seed, params.device) as device:
        try:
            model = build_backbone(inputs.numeric.shape[1], inputs.cardinalities, params).to(device)
            tensors = _input_tensors(inputs)
            targets = torch.from_numpy(labels)
            val_tensors = _input_tensors(validation[0]) if validation is not None else None
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    resolve_pos_weight(params.pos_weight, labels),
                    dtype=torch.float32,
                    device=device,
                )
            )
            optimizer_class = torch.optim.Adam if params.optimizer == "adam" else torch.optim.AdamW
            optimizer = optimizer_class(
                model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
            )
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.max_epochs)
                if params.scheduler == "cosine"
                else None
            )
            stopper = EarlyStop(params.patience, params.monitor, params.min_delta)
            rng = np.random.default_rng(seed)
            best_state = None
            for epoch in range(1, int(n_epochs) + 1):
                model.train()
                total = 0.0
                for idx in batch_indices(len(labels), params.batch_size, rng):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(tensors[0][idx].to(device), tensors[1][idx].to(device))
                    loss = criterion(logits, targets[idx].to(device))
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("nonfinite NN loss")
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("nonfinite NN gradients")
                    if params.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), params.gradient_clip_norm, error_if_nonfinite=True
                        )
                    optimizer.step()
                    total += float(loss.detach().cpu()) * len(idx)
                if any(not bool(torch.isfinite(p).all().item()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite NN weights")
                model.refresh_unknown_embeddings()
                history.append(
                    {
                        "round": epoch,
                        "dataset": "train",
                        "metric": "weighted_bce",
                        "value": total / len(labels),
                    }
                )
                if curve_metric is not None:
                    p = probabilities(_logits(model, tensors, device, params.prediction_batch_size))
                    history.append(
                        {
                            "round": epoch,
                            "dataset": "train",
                            "metric": curve_metric,
                            "value": metric_value(curve_metric, labels, p),
                        }
                    )
                if validation is not None:
                    p = probabilities(
                        _logits(model, val_tensors, device, params.prediction_batch_size)
                    )
                    score = metric_value(params.monitor, validation[1], p)
                    history.append(
                        {"round": epoch, "dataset": "val", "metric": params.monitor, "value": score}
                    )
                    if curve_metric is not None and curve_metric != params.monitor:
                        history.append(
                            {
                                "round": epoch,
                                "dataset": "val",
                                "metric": curve_metric,
                                "value": metric_value(curve_metric, validation[1], p),
                            }
                        )
                    logger.info(
                        "NN epoch %d: loss=%.6f val_%s=%.6f",
                        epoch,
                        total / len(labels),
                        params.monitor,
                        score,
                    )
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
                    raise RuntimeError("no valid NN checkpoint")
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
        },
    )


def predict_logits(model, inputs, params, seed):
    with execution_context(params, seed, params.inference_device) as device:
        try:
            model.to(device)
            return _logits(model, _input_tensors(inputs), device, params.prediction_batch_size)
        finally:
            model.cpu()
