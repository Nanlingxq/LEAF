#!/usr/bin/env python3
"""Run the fixed per-trial SSD protocol for Shin2017A on the LEAF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from init_text_embeddings import load_embeddings
from load_datasets import load_by_index
from run_shin2017a_unlearning_retraining import load_data, model_for, sha256, write_json


def now():
    return datetime.now().astimezone().isoformat()


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def checkpoint_state(path, expected_sha):
    path = Path(path)
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}: {actual}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(config, checkpoint, expected_sha, device):
    model = model_for(config).to(device)
    state = checkpoint_state(checkpoint, expected_sha)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {result}")
    embedding = torch.tensor(
        load_embeddings(config["model"]["text_embedding"])["None"],
        dtype=torch.float32,
        device=device,
    )
    return model, embedding, state


def state_equal(left, right):
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def apply_ssd_tensor(weight, retain_importance, forget_importance, alpha, dampening, exponent):
    if weight.shape != retain_importance.shape or weight.shape != forget_importance.shape:
        raise ValueError("Weight and importance tensor shapes must match")
    if not torch.isfinite(retain_importance).all() or not torch.isfinite(forget_importance).all():
        raise ValueError("Importance values must be finite")
    if (retain_importance < 0).any() or (forget_importance < 0).any():
        raise ValueError("Squared-gradient importance cannot be negative")
    selected = forget_importance > alpha * retain_importance
    scale = torch.ones_like(weight, dtype=torch.float32)
    valid = selected & (forget_importance > 0)
    ratio = torch.zeros_like(scale)
    ratio[valid] = retain_importance[valid] / forget_importance[valid]
    scale[valid] = torch.clamp(ratio[valid], max=1.0).pow(dampening * exponent)
    scale[selected & (forget_importance == 0)] = 0.0
    updated = weight * scale.to(dtype=weight.dtype)
    return updated, selected, scale


def synthetic_check():
    weight = torch.tensor([2.0, -2.0, 3.0, 4.0, 5.0, 0.0])
    retain = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    forget = torch.tensor([20.0, 10.0, 11.0, 1.0, 0.0, 20.0])
    updated, selected, scale = apply_ssd_tensor(weight, retain, forget, 10.0, 1.0, 1.0)
    expected_selected = torch.tensor([True, False, True, True, False, True])
    expected = torch.tensor([0.1, -2.0, 3.0 / 11.0, 0.0, 5.0, 0.0])
    if not torch.equal(selected, expected_selected) or not torch.allclose(updated, expected, atol=1e-7):
        raise AssertionError({"selected": selected.tolist(), "updated": updated.tolist()})
    if not torch.equal(updated[~selected], weight[~selected]):
        raise AssertionError("Unselected weights changed")
    if (updated[selected].abs() > weight[selected].abs() + 1e-7).any():
        raise AssertionError("SSD increased a selected weight magnitude")
    return {
        "passed": True,
        "selected": int(selected.sum()),
        "actually_changed": int((updated != weight).sum()),
        "strict_threshold_excluded": bool(not selected[1]),
        "both_zero_excluded": bool(not selected[4]),
        "zero_retain_importance_sets_weight_zero": bool(updated[3] == 0),
        "zero_weight_can_be_selected_without_change": bool(selected[5] and updated[5] == weight[5]),
        "scales": scale.tolist(),
    }


def input_data(config, include_test=False):
    data = load_data(config)
    train_rows = data["manifest"]["train_samples"]
    valid_rows = data["manifest"]["valid_samples"]
    target = config["dataset"]["target_subject"]
    train_target = np.asarray([row["subject"] == target for row in train_rows])
    valid_target = np.asarray([row["subject"] == target for row in valid_rows])
    result = {
        "train_x": data["train_x"],
        "train_y": data["train_y"],
        "valid_x": data["valid_x"],
        "valid_y": data["valid_y"],
        "target_train_x": data["train_x"][train_target],
        "target_train_y": data["train_y"][train_target],
        "retain_train_x": data["train_x"][~train_target],
        "retain_train_y": data["train_y"][~train_target],
        "target_valid_x": data["valid_x"][valid_target],
        "target_valid_y": data["valid_y"][valid_target],
        "retain_valid_x": data["valid_x"][~valid_target],
        "retain_valid_y": data["valid_y"][~valid_target],
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "manifest": data["manifest"],
    }
    if include_test:
        groups = config["dataset"]["test_subject_groups"]
        all_groups = config["dataset"]["all_subject_groups"]
        indices = [all_groups.index(group) for group in groups]
        test_x, test_y = load_by_index(config["dataset"]["name"], indices)
        test_rows = [
            {"subject": subject, "trial_index_zero_based": trial}
            for subject in groups
            for trial in range(60)
        ]
        if len(test_y) != config["dataset"]["test_sample_count_from_archived_split"]:
            raise ValueError(f"Unexpected test sample count: {len(test_y)}")
        result.update(test_x=test_x.astype(np.float32), test_y=test_y.astype(np.int64), test_rows=test_rows)
    return result


def model_snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def scan_importance(model, embedding, x, y, device, batch_size):
    if batch_size != 1:
        raise ValueError("The archived SSD protocol requires per-trial batch size 1")
    model.eval()
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    totals = {name: torch.zeros_like(parameter, dtype=torch.float32, device=device) for name, parameter in params}
    non_none = defaultdict(int)
    started = time.perf_counter()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    for idx in range(len(y)):
        sample = torch.from_numpy(x[idx : idx + 1]).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(y[idx : idx + 1]).to(device=device, dtype=torch.long)
        logits = model(sample, embedding)
        loss = criterion(logits.float(), labels)
        gradients = torch.autograd.grad(loss, [value for _, value in params], allow_unused=True)
        for (name, _), gradient in zip(params, gradients):
            if gradient is not None:
                totals[name].add_(gradient.detach().float().square())
                non_none[name] += 1
        if (idx + 1) % 100 == 0 or idx + 1 == len(y):
            print(f"importance {idx + 1}/{len(y)}", flush=True)
    elapsed = time.perf_counter() - started
    averaged = {name: (value / float(len(y))).detach().cpu() for name, value in totals.items()}
    for name, value in averaged.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite importance values: {name}")
    return averaged, {
        "samples": len(y),
        "elapsed_seconds": elapsed,
        "parameters_with_any_gradient": sum(count > 0 for count in non_none.values()),
        "parameters_without_gradient": sum(count == 0 for count in non_none.values()),
        "non_none_gradient_trials_by_parameter": dict(non_none),
    }


def run_preflight(config):
    data = input_data(config)
    expected = config["dataset"]["expected_counts"]
    if len(data["train_rows"]) != expected["full_train"] or len(data["valid_rows"]) != expected["full_valid"]:
        raise ValueError("Observed split does not match archived counts")
    original = config["checkpointing"]["original"]["best_retain_validation"]
    retrain = config["checkpointing"]["retrain"]["best_retain_validation"]
    original_hash = config["checkpoint_hashes"]["original"]
    retrain_hash = config["checkpoint_hashes"]["retrain"]
    original_state = checkpoint_state(original, original_hash)
    retrain_state = checkpoint_state(retrain, retrain_hash)
    device = torch.device("cpu")
    model = model_for(config)
    model.load_state_dict(original_state, strict=True)
    embedding = torch.tensor(load_embeddings(config["model"]["text_embedding"])["None"], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(data["train_x"][:1].astype(np.float32)), embedding)
    if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("Original checkpoint CPU forward failed")
    model.load_state_dict(retrain_state, strict=True)
    with torch.inference_mode():
        retrain_logits = model(torch.from_numpy(data["train_x"][:1].astype(np.float32)), embedding)
    if tuple(retrain_logits.shape) != (1, 2) or not torch.isfinite(retrain_logits).all():
        raise ValueError("Retrain checkpoint CPU forward failed")
    formula = synthetic_check()
    result = {
        "status": "passed",
        "created_at": now(),
        "device": "CPU",
        "dataset_sha256": sha256(config["dataset"]["path"]),
        "split_manifest_sha256": sha256(config["dataset"]["split_manifest_path"]),
        "observed_counts": data["manifest"]["counts"],
        "original_checkpoint_sha256": original_hash,
        "retrain_checkpoint_sha256": retrain_hash,
        "strict_load": {"original": True, "retrain": True, "missing": 0, "unexpected": 0},
        "model_forward": {
            "input_shape": list(data["train_x"][:1].shape),
            "original_logits_shape": list(logits.shape),
            "retrain_logits_shape": list(retrain_logits.shape),
            "finite": True,
        },
        "synthetic_formula": formula,
        "test_data_loaded": False,
        "optimizer_steps": 0,
    }
    write_json(config["outputs"]["preflight_json"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_importance(config):
    expected_gpu = str(config["execution"]["cuda_visible_devices"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={expected_gpu} as archived in the effective config")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device from the configured physical GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    data = input_data(config)
    # Arrays are aligned with the split manifest's train rows.
    train_target = np.asarray([row["subject"] == config["dataset"]["target_subject"] for row in data["train_rows"]])
    full_x, full_y = data["train_x"], data["train_y"]
    target_x, target_y = full_x[train_target], full_y[train_target]
    model, embedding, original_state = load_model(
        config,
        config["checkpointing"]["original"]["best_retain_validation"],
        config["checkpoint_hashes"]["original"],
        device,
    )
    before = model_snapshot(model)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"starting full-data importance scan: {len(full_y)} trials", flush=True)
    full_importance, full_stats = scan_importance(model, embedding, full_x, full_y, device, 1)
    if not state_equal(before, model_snapshot(model)):
        raise RuntimeError("Model state changed during full-data importance scan")
    print(f"starting target importance scan: {len(target_y)} trials", flush=True)
    target_importance, target_stats = scan_importance(model, embedding, target_x, target_y, device, 1)
    if not state_equal(before, model_snapshot(model)):
        raise RuntimeError("Model state changed during target importance scan")
    expected_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if full_importance.keys() != expected_names or target_importance.keys() != expected_names:
        raise RuntimeError("Importance parameter names do not match model parameters")
    out = Path(config["outputs"]["importance_directory"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite importance outputs: {out}")
    importance_path = out / "per-trial-gradient-squared-mean.pt"
    temp_path = importance_path.with_suffix(".pt.tmp")
    torch.save({"full": full_importance, "target": target_importance}, temp_path)
    temp_path.replace(importance_path)
    identity = {
        "created_at": now(),
        "original_checkpoint_sha256": config["checkpoint_hashes"]["original"],
        "dataset_sha256": config["dataset"]["sha256"],
        "split_manifest_sha256": config["split_manifest_sha256"],
        "target_subject": config["dataset"]["target_subject"],
        "loss": "CrossEntropyLoss, one trial per forward/backward",
        "precision": "float32 forward, gradient and accumulation; TF32 disabled",
        "model_mode": "eval; autograd enabled; no optimizer updates",
        "batch_size": 1,
        "parameter_shapes": {name: list(value.shape) for name, value in full_importance.items()},
        "full_scan": full_stats,
        "target_scan": target_stats,
        "test_data_loaded": False,
        "file_sha256": sha256(importance_path),
        "file_bytes": importance_path.stat().st_size,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    write_json(out / "importance-manifest.json", identity)
    print(json.dumps(identity, ensure_ascii=False, indent=2), flush=True)


def run_update(config):
    formula_cfg = config["ssd"]
    input_path = config["checkpointing"]["original"]["best_retain_validation"]
    original_hash = config["checkpoint_hashes"]["original"]
    source_state = checkpoint_state(input_path, original_hash)
    importance_path = Path(config["outputs"]["importance_file"])
    importance_manifest_path = Path(config["outputs"]["importance_manifest"])
    importance_manifest = json.loads(importance_manifest_path.read_text(encoding="utf-8"))
    if sha256(importance_path) != importance_manifest["file_sha256"]:
        raise ValueError("Importance artifact SHA256 mismatch")
    importance = torch.load(importance_path, map_location="cpu", weights_only=True)
    model = model_for(config)
    model.load_state_dict(source_state, strict=True)
    parameter_names = {name for name, _ in model.named_parameters()}
    if importance["full"].keys() != parameter_names or importance["target"].keys() != parameter_names:
        raise ValueError("Importance parameter names do not exactly match model parameter names")
    updated = {key: value.clone() for key, value in source_state.items()}
    stats = defaultdict(lambda: {"elements": 0, "selected": 0, "changed": 0, "l2_delta_sq": 0.0, "weight_l2_sq": 0.0, "scales": []})
    all_selected = 0
    all_changed = 0
    for name, _ in model.named_parameters():
        original = source_state[name]
        retain_imp = importance["full"][name]
        target_imp = importance["target"][name]
        if original.shape != retain_imp.shape or original.shape != target_imp.shape:
            raise ValueError(f"Shape mismatch for {name}")
        new_value, selected, scale = apply_ssd_tensor(
            original,
            retain_imp.float(),
            target_imp.float(),
            float(formula_cfg["selection_weighting"]),
            float(formula_cfg["dampening_constant"]),
            float(formula_cfg["exponent"]),
        )
        changed = new_value != original
        if not torch.equal(new_value[~selected], original[~selected]):
            raise RuntimeError(f"Unselected values changed: {name}")
        if (new_value[selected].abs() > original[selected].abs() + 1e-7).any():
            raise RuntimeError(f"Selected weight magnitude increased: {name}")
        if not torch.isfinite(new_value).all():
            raise FloatingPointError(f"Updated parameter is non-finite: {name}")
        updated[name] = new_value
        layer = name.split(".", 1)[0]
        row = stats[layer]
        row["elements"] += original.numel()
        row["selected"] += int(selected.sum())
        row["changed"] += int(changed.sum())
        row["l2_delta_sq"] += float((new_value.float() - original.float()).square().sum())
        row["weight_l2_sq"] += float(original.float().square().sum())
        selected_scales = scale[selected]
        if selected_scales.numel():
            row["scales"].append(selected_scales)
        all_selected += int(selected.sum())
        all_changed += int(changed.sum())
    buffers = [name for name in source_state if name not in parameter_names]
    if any(not torch.equal(updated[name], source_state[name]) for name in buffers):
        raise RuntimeError("A non-parameter buffer changed")
    output_path = Path(config["outputs"]["ssd_checkpoint"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite SSD checkpoint: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(updated, temporary)
    temporary.replace(output_path)
    layer_results = {}
    for layer, row in stats.items():
        scales = torch.cat(row.pop("scales")) if row["scales"] else torch.empty(0)
        layer_results[layer] = {
            **row,
            "selected_fraction": row["selected"] / row["elements"],
            "changed_fraction": row["changed"] / row["elements"],
            "relative_l2_change": (row["l2_delta_sq"] / max(row["weight_l2_sq"], 1e-30)) ** 0.5,
            "selected_scale_min": float(scales.min()) if scales.numel() else None,
            "selected_scale_median": float(scales.median()) if scales.numel() else None,
            "selected_scale_max": float(scales.max()) if scales.numel() else None,
        }
    result = {
        "created_at": now(),
        "status": "completed",
        "original_checkpoint": input_path,
        "original_checkpoint_sha256": original_hash,
        "importance_file": str(importance_path),
        "importance_sha256": sha256(importance_path),
        "ssd_checkpoint": str(output_path),
        "ssd_checkpoint_sha256": sha256(output_path),
        "ssd_checkpoint_bytes": output_path.stat().st_size,
        "selection_weighting": formula_cfg["selection_weighting"],
        "dampening_constant": formula_cfg["dampening_constant"],
        "exponent": formula_cfg["exponent"],
        "selected_parameter_elements": all_selected,
        "actually_changed_parameter_elements": all_changed,
        "state_dict_buffers_unchanged": len(buffers),
        "all_finite": True,
        "test_data_loaded": False,
        "layers": layer_results,
    }
    write_json(config["outputs"]["update_json"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def metric_summary(y_true, logits):
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "n": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "auroc": float(roc_auc_score(y_true, probabilities[:, 1])),
        "recall_class_0": float(recall_score(y_true, predictions, labels=[0], average="macro", zero_division=0)),
        "recall_class_1": float(recall_score(y_true, predictions, labels=[1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "cross_entropy_loss": float(F.cross_entropy(torch.tensor(logits, dtype=torch.float64), torch.tensor(y_true, dtype=torch.long))),
    }, probabilities


def predict(model, embedding, x, device, batch_size=32):
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            values = torch.from_numpy(x[start : start + batch_size]).to(device=device, dtype=torch.float32)
            logits.append(model(values, embedding).float().cpu().numpy())
    return np.concatenate(logits, axis=0)


def js_divergence(prob_a, prob_b):
    middle = 0.5 * (prob_a + prob_b)
    log_a = np.log(np.clip(prob_a, 1e-12, 1.0))
    log_b = np.log(np.clip(prob_b, 1e-12, 1.0))
    log_m = np.log(np.clip(middle, 1e-12, 1.0))
    return 0.5 * np.sum(prob_a * (log_a - log_m), axis=1) + 0.5 * np.sum(prob_b * (log_b - log_m), axis=1)


def run_evaluation(config):
    expected_gpu = str(config["execution"]["cuda_visible_devices"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={expected_gpu} as archived in the effective config")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device from the configured physical GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    eval_started = time.perf_counter()
    data = input_data(config, include_test=True)
    partitions = {
        "test": (data["test_x"], data["test_y"], data["test_rows"]),
        "target_train": (data["target_train_x"], data["target_train_y"], [r for r in data["train_rows"] if r["subject"] == config["dataset"]["target_subject"]]),
        "target_valid": (data["target_valid_x"], data["target_valid_y"], [r for r in data["valid_rows"] if r["subject"] == config["dataset"]["target_subject"]]),
        "retain_train": (data["retain_train_x"], data["retain_train_y"], [r for r in data["train_rows"] if r["subject"] != config["dataset"]["target_subject"]]),
        "retain_valid": (data["retain_valid_x"], data["retain_valid_y"], [r for r in data["valid_rows"] if r["subject"] != config["dataset"]["target_subject"]]),
    }
    checkpoint_refs = {
        "original": (config["checkpointing"]["original"]["best_retain_validation"], config["checkpoint_hashes"]["original"]),
        "retrain": (config["checkpointing"]["retrain"]["best_retain_validation"], config["checkpoint_hashes"]["retrain"]),
        "ssd": (config["outputs"]["ssd_checkpoint"], config["checkpoint_hashes"]["ssd"]),
    }
    output_dir = Path(config["outputs"]["evaluation_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "predictions.npz"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir / name}")
    results = {"models": {}, "pairwise": {}, "mia": {}}
    predictions = {}
    gpu_costs = {}
    torch.cuda.reset_peak_memory_stats(device)
    for name, (path, digest) in checkpoint_refs.items():
        model_started = time.perf_counter()
        model, embedding, _ = load_model(config, path, digest, device)
        load_seconds = time.perf_counter() - model_started
        model.eval()
        logits_by_part = {}
        metrics = {}
        model_predict_started = time.perf_counter()
        for part, (x, y, rows) in partitions.items():
            logits = predict(model, embedding, x, device)
            summary, probabilities = metric_summary(y, logits)
            metrics[part] = summary
            logits_by_part[part] = logits
            predictions[f"{part}_y"] = y
            predictions[f"{part}_subjects"] = np.asarray([row["subject"] for row in rows], dtype="U3")
            predictions[f"{part}_trial_indices"] = np.asarray([row["trial_index_zero_based"] for row in rows], dtype=np.int16)
            predictions[f"{part}_{name}_logits"] = logits
            predictions[f"{part}_{name}_probabilities"] = probabilities.astype(np.float32)
        prediction_seconds = time.perf_counter() - model_predict_started
        test_y = partitions["test"][1]
        test_rows = partitions["test"][2]
        per_subject = {}
        for subject in config["dataset"]["test_subject_groups"]:
            mask = np.asarray([row["subject"] == subject for row in test_rows])
            per_subject[subject] = metric_summary(test_y[mask], logits_by_part["test"][mask])[0]
        equal_subject_means = {
            metric_name: float(np.mean([item[metric_name] for item in per_subject.values()]))
            for metric_name in ("balanced_accuracy", "cohen_kappa", "accuracy", "weighted_f1", "auroc")
        }
        results["models"][name] = {
            "checkpoint": path,
            "checkpoint_sha256": digest,
            "metrics": metrics,
            "test_per_subject": per_subject,
            "test_equal_subject_means": equal_subject_means,
        }

        # Fit a separate, fixed-parameter entropy attack for each model.
        retain_train_probs = torch.softmax(torch.tensor(logits_by_part["retain_train"], dtype=torch.float64), dim=1).numpy()
        retain_valid_probs = torch.softmax(torch.tensor(logits_by_part["retain_valid"], dtype=torch.float64), dim=1).numpy()
        target_train_probs = torch.softmax(torch.tensor(logits_by_part["target_train"], dtype=torch.float64), dim=1).numpy()
        target_valid_probs = torch.softmax(torch.tensor(logits_by_part["target_valid"], dtype=torch.float64), dim=1).numpy()
        entropy = lambda p: -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1).reshape(-1, 1)
        attack_x = np.concatenate([entropy(retain_train_probs), entropy(retain_valid_probs)])
        attack_y = np.concatenate([np.ones(len(retain_train_probs)), np.zeros(len(retain_valid_probs))]).astype(int)
        attack = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)
        attack.fit(attack_x, attack_y)
        target_x = np.concatenate([entropy(target_train_probs), entropy(target_valid_probs)])
        target_y = np.concatenate([np.ones(len(target_train_probs)), np.zeros(len(target_valid_probs))]).astype(int)
        target_score = attack.predict_proba(target_x)[:, 1]
        target_prediction = (target_score >= 0.5).astype(int)
        results["mia"][name] = {
            "feature": "predictive entropy",
            "train_members": int(len(retain_train_probs)),
            "validation_nonmembers": int(len(retain_valid_probs)),
            "target_membership_label_basis": "historical Original task-training membership, including for retrain",
            "target_train_member_prediction_rate": float(target_prediction[: len(target_train_probs)].mean()),
            "target_combined_roc_auc": float(roc_auc_score(target_y, target_score)),
            "target_combined_balanced_accuracy": float(balanced_accuracy_score(target_y, target_prediction)),
            "target_train_member_positive_rate": float(target_prediction[: len(target_train_probs)].mean()),
            "target_valid_nonmember_positive_rate": float(target_prediction[len(target_train_probs) :].mean()),
            "positive_rate_by_true_class": {
                "label_0": float(target_prediction[target_y == 0].mean()),
                "label_1": float(target_prediction[target_y == 1].mean()),
            },
            "exploratory_only": True,
        }
        gpu_costs[name] = {
            "load_seconds": load_seconds,
            "evaluation_seconds": prediction_seconds,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
        del model, embedding
        torch.cuda.empty_cache()

    target_comparisons = {}
    for part in ("target_train", "target_valid"):
        o = torch.softmax(torch.tensor(predictions[f"{part}_original_logits"], dtype=torch.float64), dim=1).numpy()
        r = torch.softmax(torch.tensor(predictions[f"{part}_retrain_logits"], dtype=torch.float64), dim=1).numpy()
        s = torch.softmax(torch.tensor(predictions[f"{part}_ssd_logits"], dtype=torch.float64), dim=1).numpy()
        target_comparisons[part] = {}
        for left, right, p_left, p_right in (("original", "retrain", o, r), ("ssd", "retrain", s, r)):
            key = f"{left}_vs_{right}"
            target_comparisons[part][key] = {
                "prediction_disagreement_rate": float(np.mean(p_left.argmax(axis=1) != p_right.argmax(axis=1))),
                "mean_js_divergence_nats": float(js_divergence(p_left, p_right).mean()),
                "per_sample_js_divergence_nats": js_divergence(p_left, p_right).tolist(),
            }
    results["pairwise"]["target_probability_closeness"] = target_comparisons
    results["costs"] = gpu_costs
    results["elapsed_seconds"] = time.perf_counter() - eval_started
    results["test_data_loaded"] = True
    results["test_data_sha256"] = config["dataset"]["sha256"]
    results["evaluation_protocol"] = config["evaluation"]
    pred_path = output_dir / "predictions.npz"
    temp_pred = pred_path.with_suffix(".npz.tmp")
    with temp_pred.open("wb") as stream:
        np.savez_compressed(stream, **predictions)
    temp_pred.replace(pred_path)
    results["prediction_archive"] = str(pred_path)
    results["prediction_archive_sha256"] = sha256(pred_path)
    results["prediction_archive_bytes"] = pred_path.stat().st_size
    comparison_path = output_dir / "comparison.json"
    write_json(comparison_path, results)
    write_json(
        output_dir / "evaluation-artifacts.json",
        {
            "comparison_json": str(comparison_path),
            "comparison_json_sha256": sha256(comparison_path),
            "prediction_archive": str(pred_path),
            "prediction_archive_sha256": sha256(pred_path),
            "created_at": now(),
        },
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["preflight", "importance", "update", "evaluate"])
    args = parser.parse_args()
    config = read_config(args.config)
    stage_config = config.get("ssd_experiment", {})
    if stage_config.get("stage") != args.stage:
        raise ValueError(f"Config stage {stage_config.get('stage')} != requested {args.stage}")
    if args.stage == "preflight":
        run_preflight(config)
    elif args.stage == "importance":
        run_importance(config)
    elif args.stage == "update":
        run_update(config)
    else:
        run_evaluation(config)


if __name__ == "__main__":
    main()
