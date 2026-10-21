#!/usr/bin/env python3
"""Train and archive the two Shin2017A models required before SSD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d_downstream import DownStream, eval_model
from init_text_embeddings import load_embeddings
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml
from load_datasets import load_by_index, split_train_valid
from utils import cosine_annealing_lr, make_lr_lambda, seed_all


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def label_counts(labels):
    values, sizes = np.unique(labels, return_counts=True)
    return {str(int(value)): int(size) for value, size in zip(values, sizes)}


def load_data(config):
    d = config["dataset"]
    path = Path(d["path"]).resolve()
    if not path.is_file() or path.parent != Path(DATA_FOLDER).resolve():
        raise FileNotFoundError(f"Dataset path does not match LEAF loader: {path}")
    with h5py.File(path, "r", locking=False) as h5:
        groups = list(h5.keys())
        if groups != d["all_subject_groups"]:
            raise ValueError(f"Unexpected HDF5 group order: {groups}")
        if groups[:22] != d["trainval_subject_groups"] or groups[22:] != d["test_subject_groups"]:
            raise ValueError("Subject groups differ from the archived train/test split")
        records = []
        position = 0
        for subject in groups[:22]:
            group = h5[subject]
            if list(group["X"].shape) != d["subject_trial_shape"]:
                raise ValueError(f"Unexpected X shape for {subject}")
            for trial, label in enumerate(group["Y"][:].astype(np.int64)):
                records.append({
                    "subject": subject,
                    "trial_index_zero_based": trial,
                    "original_role": "valid" if position % 5 == 0 else "train",
                    "label": int(label),
                })
                position += 1

    x, y = load_by_index(d["name"], range(22))
    train_x, train_y, valid_x, valid_y = split_train_valid(x, y, val_ratio=0.2)
    train_rows = [row for row in records if row["original_role"] == "train"]
    valid_rows = [row for row in records if row["original_role"] == "valid"]
    if not np.array_equal(train_y, [row["label"] for row in train_rows]):
        raise ValueError("Training labels do not match subject/trial identities")
    if not np.array_equal(valid_y, [row["label"] for row in valid_rows]):
        raise ValueError("Validation labels do not match subject/trial identities")

    target = d["target_subject"]
    keep_train = np.array([row["subject"] != target for row in train_rows])
    keep_valid = np.array([row["subject"] != target for row in valid_rows])
    observed = {
        "full_train": len(train_y),
        "target_train": int((~keep_train).sum()),
        "retain_train": int(keep_train.sum()),
        "full_valid": len(valid_y),
        "target_valid": int((~keep_valid).sum()),
        "retain_valid": int(keep_valid.sum()),
        "full_train_labels": label_counts(train_y),
        "target_train_labels": label_counts(train_y[~keep_train]),
        "retain_train_labels": label_counts(train_y[keep_train]),
        "full_valid_labels": label_counts(valid_y),
        "target_valid_labels": label_counts(valid_y[~keep_valid]),
        "retain_valid_labels": label_counts(valid_y[keep_valid]),
    }
    if observed != d["expected_counts"]:
        raise ValueError(f"Split/count check failed: {observed}")

    manifest = {
        "dataset": d["name"], "path": str(path), "sha256": sha256(path),
        "bytes": path.stat().st_size, "trainval_subject_groups": d["trainval_subject_groups"],
        "test_subject_groups": d["test_subject_groups"], "target_subject": target,
        "labels": {"0": "Left", "1": "Right"},
        "split_rule": "Concatenate the first 22 HDF5 groups; every fifth trial is validation.",
        "test_data": {"n": d["test_sample_count_from_archived_split"], "loaded": False},
        "counts": observed, "train_samples": train_rows, "valid_samples": valid_rows,
    }
    split_path = Path(d["split_manifest_path"])
    if split_path.exists():
        previous = json.loads(split_path.read_text(encoding="utf-8"))
        if previous.get("sha256") != manifest["sha256"] or previous.get("counts") != observed:
            raise FileExistsError(f"Refusing to replace changed split manifest: {split_path}")
    else:
        write_json(split_path, manifest)

    return {
        "train_x": train_x.astype(np.float32), "train_y": train_y.astype(np.int64),
        "valid_x": valid_x.astype(np.float32), "valid_y": valid_y.astype(np.int64),
        "retain_train_x": train_x[keep_train].astype(np.float32),
        "retain_train_y": train_y[keep_train].astype(np.int64),
        "retain_valid_x": valid_x[keep_valid].astype(np.float32),
        "retain_valid_y": valid_y[keep_valid].astype(np.int64),
        "manifest": manifest,
    }


def model_for(config):
    m = config["model"]
    source = load_yaml(Path(m["source_model_config"]))
    model_config = build_model_config(source, dim=m["dimensions"], emb=m["text_embedding"])
    return DownStream(model_config, n_classes=2, trialTime=4)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def prepare(config):
    data = load_data(config)
    input_path = Path(config["input_checkpoint"]["path"])
    if sha256(input_path) != config["input_checkpoint"]["sha256"]:
        raise ValueError("Fixed instruction checkpoint SHA256 changed")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = model_for(config)
    source_state = torch.load(input_path, map_location="cpu", weights_only=True)
    model.leaf.load_state_dict(source_state, strict=True)
    embedding = torch.tensor(load_embeddings(config["model"]["text_embedding"])["None"])
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(data["train_x"][:1]), embedding)
    if list(logits.shape) != [1, 2] or not torch.isfinite(logits).all():
        raise ValueError(f"CPU forward check failed: {list(logits.shape)}")

    init_path = Path(config["shared_initialization"]["path"])
    if init_path.exists():
        raise FileExistsError(f"Refusing to overwrite common initialization: {init_path}")
    init_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(model), init_path)
    split_path = Path(config["dataset"]["split_manifest_path"])
    result = {
        "run_id": config["run_id"], "created_at": datetime.now().astimezone().isoformat(),
        "input_checkpoint_sha256": sha256(input_path),
        "strict_load": {"ok": True, "LEAF_keys": len(source_state), "missing": 0, "unexpected": 0},
        "shared_initialization": {
            "path": str(init_path), "sha256": sha256(init_path),
            "bytes": init_path.stat().st_size,
            "state_dict_entries_including_classifier": len(model.state_dict()),
        },
        "model_forward": {
            "device": "CPU", "input_shape": list(data["train_x"][:1].shape),
            "logits_shape": list(logits.shape), "finite": True, "optimizer_steps": 0,
        },
        "split_manifest": {"path": str(split_path), "sha256": sha256(split_path)},
        "observed_counts": data["manifest"]["counts"],
        "test_data_loaded": False,
    }
    write_json(config["outputs"]["preflight_json"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_state(model), temporary)
    temporary.replace(path)


def save_metrics(path, rows):
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(config, stage):
    if config["stage"]["name"] != stage:
        raise ValueError(f"Config stage {config['stage']['name']} != requested {stage}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["execution"]["cuda_visible_devices"]):
        raise ValueError("CUDA_VISIBLE_DEVICES must match the archived configuration")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected physical GPU1 visible as the only CUDA device")
    data = load_data(config)
    seed_all(42)

    if stage == "original":
        train_x, train_y = data["train_x"], data["train_y"]
    else:
        train_x, train_y = data["retain_train_x"], data["retain_train_y"]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=128, shuffle=True, num_workers=0, pin_memory=False, drop_last=False,
    )
    retain_loader = DataLoader(
        TensorDataset(torch.from_numpy(data["retain_valid_x"]), torch.from_numpy(data["retain_valid_y"])),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
    )
    full_loader = None
    if stage == "original":
        full_loader = DataLoader(
            TensorDataset(torch.from_numpy(data["valid_x"]), torch.from_numpy(data["valid_y"])),
            batch_size=128, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
        )
    if len(train_loader) != config["training"]["steps_per_epoch"][stage]:
        raise ValueError(f"Unexpected updates per epoch: {len(train_loader)}")

    out = Path(config["outputs"][stage + "_directory"])
    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite stage outputs: {out}")
    device = torch.device("cuda:0")
    state_path = out / "runner-state.json"
    state = {
        "run_id": config["run_id"], "stage": stage, "status": "running",
        "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
        "train_samples": len(train_y), "retain_valid_samples": len(data["retain_valid_y"]),
        "steps_per_epoch": len(train_loader), "test_data_loaded": False,
        "physical_gpu": 1, "logical_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    write_json(state_path, state)
    try:
        model = model_for(config)
        init_path = Path(config["shared_initialization"]["path"])
        if sha256(init_path) != config["shared_initialization"]["sha256"]:
            raise ValueError("Common initialization SHA256 changed")
        model.load_state_dict(
            torch.load(init_path, map_location="cpu", weights_only=True), strict=True
        )
        model = model.to(device)
        embedding = torch.tensor(load_embeddings(config["model"]["text_embedding"])["None"]).to(device)
        fast = list(model.leaf.tower.tokenizer.parameters()) + list(model.classifier.parameters())
        fast_ids = {id(parameter) for parameter in fast}
        slow = [parameter for parameter in model.parameters() if id(parameter) not in fast_ids]
        opt = config["training"]["optimizer"]
        optimizer = optim.AdamW(
            [
                {"params": fast, "lr": opt["fast_learning_rate"]},
                {"params": slow, "lr": opt["slow_learning_rate"]},
            ],
            betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"],
        )
        steps = len(train_loader)
        scheduler_values = cosine_annealing_lr(1.0, 0.1, 200, steps, warmup_epochs=1)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(scheduler_values))
        criterion = nn.CrossEntropyLoss()
        best_retain, best_full = -1.0, -1.0
        best_retain_epoch = best_full_epoch = None
        rows = []
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(200):
            model.train()
            total_loss, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    logits = model(x, embedding)
                    loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.item()) * int(y.numel())
                seen += int(y.numel())

            retain = eval_model(model, retain_loader, device, 2, instrct_emb=embedding)
            row = {
                "epoch": epoch + 1, "train_loss": total_loss / seen,
                "retain_val_bacc": retain[0], "retain_val_auroc": retain[1],
                "retain_val_weighted_f1": retain[2], "retain_val_kappa": retain[3],
                "full_val_bacc": "", "full_val_auroc": "",
                "full_val_weighted_f1": "", "full_val_kappa": "",
                "elapsed_seconds": time.perf_counter() - started,
            }
            if retain[0] > best_retain:
                best_retain, best_retain_epoch = retain[0], epoch + 1
                save_checkpoint(model, out / "best-retain-val.ckpt")
            if full_loader is not None:
                full = eval_model(model, full_loader, device, 2, instrct_emb=embedding)
                row.update({
                    "full_val_bacc": full[0], "full_val_auroc": full[1],
                    "full_val_weighted_f1": full[2], "full_val_kappa": full[3],
                })
                if full[0] > best_full:
                    best_full, best_full_epoch = full[0], epoch + 1
                    save_checkpoint(model, out / "best-v2-val.ckpt")
            rows.append(row)
            save_metrics(out / "metrics.csv", rows)
            print(
                f"{epoch + 1}/200 {stage} retain-val-bAcc={retain[0]:.4f} "
                f"retain-val-Kappa={retain[3]:.4f} train-loss={row['train_loss']:.6f}",
                flush=True,
            )
            if epoch == 199:
                save_checkpoint(model, out / "epoch-200.ckpt")

        state.update({
            "status": "completed", "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "best_retain_val_epoch": best_retain_epoch, "best_retain_val_bacc": best_retain,
            "best_full_val_epoch": best_full_epoch,
            "best_full_val_bacc": best_full if best_full_epoch is not None else None,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "checkpoints": {p.name: sha256(p) for p in out.glob("*.ckpt")},
            "test_data_loaded": False,
        })
        write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        state.update({
            "status": "failed", "finished_at": datetime.now().astimezone().isoformat(),
            "error_type": type(error).__name__, "error": str(error),
        })
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "original", "retrain"], required=True)
    args = parser.parse_args()
    config = read_config(args.config.resolve())
    if args.stage == "prepare":
        prepare(config)
    else:
        train(config, args.stage)


if __name__ == "__main__":
    main()
