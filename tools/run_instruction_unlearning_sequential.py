"""Preparation stages for instruction-level SSD/LAF experiments.

The LEAF implementation and instruction training module are loaded from an
immutable historical snapshot. Their CBCR License 1.0 remains in that snapshot.
This entry point does not load held-out EEG or test arrays during preparation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_save(torch, value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def gpu_inventory():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.total",
        "--format=csv,noheader,nounits"], text=True)
    return [{"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
             "free_mib": int(row[3]), "total_mib": int(row[4])}
            for row in csv.reader(raw.splitlines())]


def runtime(config):
    work = Path(config["workspace"])
    sys.path.insert(0, str(work))
    os.chdir(work)
    import numpy as np
    import torch
    import yaml
    module = importlib.import_module("b_instruct_tuning")
    torch.set_num_threads(8)
    return np, torch, yaml, module


def task_family(ds):
    for prefix, family in (("MI", "Motor"), ("EMO", "Emotion"),
                           ("CS", "Speech"), ("SSVEP", "SSVEP")):
        if ds.startswith(prefix):
            return family
    return ds


def build_index(config):
    """Reproduce historical splits before filtering any target rows."""
    import h5py
    import numpy as np
    result = {"train": [], "valid": [], "datasets": [], "test_arrays_loaded": False}
    for spec in config["datasets"]:
        ds, path = spec["dataset"], Path(spec["path"])
        st = path.stat()
        old = spec["file_identity"]
        if st.st_size != old["bytes"] or st.st_mtime_ns != old["mtime_ns"]:
            raise RuntimeError(f"Historical dataset file identity changed: {ds}")
        counts = {}
        with h5py.File(path, "r", locking=False) as f:
            split = spec["split"]
            if "trainX" in f:
                for role, xkey, ykey in (("train", "trainX", "trainY"),
                                          ("valid", "validX", "validY")):
                    labels = np.asarray(f[ykey][:]).reshape(-1)
                    for i, y in enumerate(labels):
                        result[role].append([ds, xkey, i, int(y), "predefined"])
                    counts[role] = len(labels)
            else:
                if ds == "CS_BCIC_Speech":
                    groups = list(f.keys())
                    for role, low, high in (("train", 0, 250), ("valid", 250, 300)):
                        n = 0
                        for subject in groups:
                            for i in range(low, high):
                                y = int(np.asarray(f[subject]["Y"][i]).item())
                                result[role].append([ds, subject + "/X", i, y, subject])
                                n += 1
                        counts[role] = n
                else:
                    groups = [str(x) for x in split["trainval_groups"]]
                    counter = 0
                    counts = {"train": 0, "valid": 0}
                    for subject in groups:
                        group = f[subject]
                        for i in range(len(group["X"])):
                            # The target validation labels stay unread until sealed evaluation.
                            target_holdout = ds == "MI_Shin2017A" and subject == "S01" and i % 5 == 0
                            y = -1 if target_holdout else int(np.asarray(group["Y"][i]).item())
                            if ds == "MI_HighGamma" and y == 2:
                                continue
                            role = "valid" if counter % 5 == 0 else "train"
                            result[role].append([ds, subject + "/X", i, y, subject])
                            counts[role] += 1
                            counter += 1
            for role in ("train", "valid"):
                if counts[role] != spec["split"][role]["n"]:
                    raise RuntimeError(f"Split mismatch {ds}/{role}: {counts[role]}")
        print(f"Data identity and split: {ds} {counts}", flush=True)
        # Byte hashing audits file identity; it does not decode or expose test arrays.
        result["datasets"].append({"dataset": ds, "path": str(path),
                                   "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                                   "sha256": digest(path), "counts": counts})
    assert len(result["train"]) == 167091 and len(result["valid"]) == 61163
    for subject in ("S01", "S02", "S03"):
        assert sum(x[0] == "MI_Shin2017A" and x[4] == subject for x in result["train"]) == 48
    return result


def initialize(args):
    import yaml
    root, run = Path(args.root).resolve(), Path(args.run_dir).resolve()
    if run.exists():
        raise FileExistsError(run)
    if run.parent != root / ".codex/runs":
        raise ValueError("run-dir must be a direct child of .codex/runs")
    artifact = root / "artifacts" / run.name
    history = root / "artifacts/v1-shin-pretrain-instruct-mpnet-seed42/workspace"
    historical_config = root / ".codex/runs/v1-shin-pretrain-instruct-mpnet-seed42/configs/instruction.yaml"
    source = artifact / "workspace"
    for d in (run / "configs/source", run / "logs", run / "checks", artifact, source):
        d.mkdir(parents=True, exist_ok=True)
    for file in history.glob("*.py"):
        shutil.copy2(file, source / file.name)
    for folder in ("configs", "leaf_datasets"):
        shutil.copytree(history / folder, source / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "text_embeddings").mkdir()
    for file in (history / "text_embeddings").glob("mpnet-base.*"):
        shutil.copy2(file, source / "text_embeddings" / file.name)
    shutil.copy2(root / "LICENSE", source / "LICENSE")
    shutil.copy2(historical_config, run / "configs/source/historical-instruction.yaml")
    for file in (source / "configs").glob("*.yaml"):
        shutil.copy2(file, run / "configs/source" / file.name)
    plan = root / ".codex/plans/unLearning-plan-V1-SSD+LAF_test.md"
    shutil.copy2(plan, run / "configs/source/authorized-plan.md")
    shutil.copy2(Path(__file__), artifact / "run_instruction_unlearning.py")
    common = yaml.safe_load(historical_config.read_text())
    common.update(schema_version=1, run_id=run.name, attempt=1, created_at=now(),
                  status="prepared", plan_path=str(plan), plan_sha256=digest(plan),
                  workspace=str(source), run_dir=str(run), artifact_dir=str(artifact),
                  initial_state=str(artifact / "initial/full-initial.ckpt"),
                  original_checkpoint=str(history / "checkpoints/LEAF_mpnet-42-leaf-v1.0-pretrain/it100.ckpt"),
                  original_sha256="d44baeedd580110b849d9450b7475bab78d0aec7eea87e07d75a2c99f5dcbf85",
                  index_path=str(run / "splits/index.json.gz"), excluded_subjects=["S01"],
                  temperature={"enabled": False, "logit_scale": 1.0},
                  initialization_note="Reconstructed from historical source and seed; historical initial state was not saved, so equality to it cannot be proved.")
    common["authorization"] = {
        "date": "2026-10-25", "scope": "Execute the complete SSD+LAF V1 plan in order",
        "gpu_policy": "One GPU at a time, choose any GPU with sufficient free memory; sharing allowed",
        "oom_wait_seconds": 1200, "resource_poll_seconds": 60, "inspection_interval_seconds": 3600}
    common["execution"] = {"python": sys.executable, "script": str(artifact / "run_instruction_unlearning.py"),
                           "cwd": str(source), "physical_gpu": "resolved separately for each attempt",
                           "logical_gpu": "cuda:0", "CUDA_VISIBLE_DEVICES": "selected GPU UUID",
                           "environment_overrides": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                                                     "PYTHONUNBUFFERED": "1"}}
    common["training"]["batch_per_gpu"] = 256
    common["training"]["effective_batch"] = 256
    common["evaluation"] = {"valid_every_epochs": 10, "test_evaluation": False,
                            "target_validation_access": False, "checkpoint": "fixed epoch 100"}
    common["checkpointing"] = {"weights_every_epochs": 10, "fixed_final_epoch": 100,
                               "full_resume_every_epochs": 1,
                               "resume_note": "Restore optimizer/scheduler/loops and main RNG at epoch boundary; worker RNG is not guaranteed bitwise reproducible across retries."}
    common["source_snapshot"] = {"workspace": str(source), "historical_workspace": str(history)}
    common["outputs"] = {"artifact_dir": str(artifact)}
    common["tricks"]["test_data"] = "No test arrays loaded; explicit indexed train/valid reads"
    for stage in ("prepare", "selfcheck", "preflight", "retrain", "importance", "vae"):
        c = copy.deepcopy(common)
        c["stage"] = stage
        c["stage_settings"] = {
            "prepare": {"audit": "historical metadata, per-dataset counts and complete byte SHA256"},
            "selfcheck": {"synthetic_only": True, "checks": ["masked Adam with projection and state clearing", "VAE input gradient"]},
            "preflight": {"input": [65, 2000], "precision": "float32", "mode": "eval", "prototypes": 33},
            "retrain": {"exclude_train": 48, "exclude_valid": 12, "train_count": 167043, "valid_count": 61151,
                        "epochs": 100, "batch": 256, "workers": 4, "resume": "last complete epoch"},
            "importance": {"overall_count": 167091, "subject_counts": {"S01": 48, "S02": 48, "S03": 48},
                           "batch": 1, "precision": "float32", "mode": "eval", "loss": "33-class CE, scale1",
                           "view": "first task instruction blended with normalized candidate mean; deterministic right-pad/crop2000",
                           "alpha": 1.0, "lambda": 1.0, "exponent": 1.0,
                           "checkpoint_every_samples": 1000, "workers": 0},
            "vae": {"dimensions": [768, 256, 64, 16], "decoder": [16, 64, 256, 768],
                    "activation": "ReLU; linear output", "epochs_full": 20, "epochs_target": 50,
                    "batch_full": 128, "batch_target": 16, "optimizer": "Adam", "lr": 0.001,
                    "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "precision": "float32",
                    "kl_coefficient": 0.025, "logvar_clamp": [-10, 10], "selection": "fixed final epoch"}
        }[stage]
        c["execution"]["command"] = f"{sys.executable} -u '{artifact / 'run_instruction_unlearning.py'}' --run-dir '{run}' {stage}"
        (run / "configs" / (stage + ".yaml")).write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
    index = build_index(common)
    (run / "splits").mkdir()
    with gzip.open(common["index_path"], "wt") as f:
        json.dump(index, f, separators=(",", ":"))
    common["datasets"] = [dict(spec, current_file_identity=index["datasets"][i]) for i, spec in enumerate(common["datasets"])]
    for p in (run / "configs").glob("*.yaml"):
        c = yaml.safe_load(p.read_text())
        c["datasets"] = common["datasets"]
        c["split_manifest"] = {"path": common["index_path"], "sha256": digest(common["index_path"]),
                               "train": 167091, "valid": 61163, "target_valid_labels": "unread sentinel -1"}
        c["source_configs"] = [{"path": str(q), "sha256": digest(q)} for q in (run / "configs/source").rglob("*") if q.is_file()]
        p.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
    source_manifest = [{"path": str(p.relative_to(source)), "sha256": digest(p)} for p in source.rglob("*") if p.is_file()]
    write_json(run / "checks/source-manifest.json", source_manifest)
    write_json(run / "manifest.json", {"run_id": run.name, "status": "prepared", "created_at": now(),
               "authorization": common["authorization"], "plan_sha256": digest(plan),
               "configs": [{"path": str(p), "sha256": digest(p)} for p in (run / "configs").glob("*.yaml")],
               "index_sha256": digest(common["index_path"]), "artifact_dir": str(artifact),
               "source_manifest_sha256": digest(run / "checks/source-manifest.json")})
    print("Initialization complete", run, flush=True)


class DataAccess:
    """Lazy, per-process HDF5 access; no test array access during preparation."""

    def __init__(self, config, records, text2emb, targets, instructions, key2idx, training=False, fixed=True):
        self.paths = {s["dataset"]: s["path"] for s in config["datasets"]}
        self.records = records
        self.text2emb, self.targets = text2emb, targets
        self.instructions, self.key2idx = instructions, key2idx
        self.training, self.fixed = training, fixed
        self.handles = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import h5py
        import numpy as np
        ds, xkey, row, y, _ = self.records[index]
        pid_key = (os.getpid(), ds)
        if pid_key not in self.handles:
            self.handles[pid_key] = h5py.File(self.paths[ds], "r", locking=False)
        x = np.asarray(self.handles[pid_key][xkey][row], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite EEG: {self.records[index]}")
        if self.fixed:
            text = self.instructions[task_family(ds)][0]
            blend = True
        else:
            text = random.choice(self.instructions["Default"] if random.random() < .1 else self.instructions[task_family(ds)])
            blend = random.random() < .9
        vec = self.text2emb[text]
        if blend:
            candidate = np.mean([self.text2emb[t] for t in self.targets[ds]], axis=0)
            candidate = candidate / np.linalg.norm(candidate)
            vec = (vec + candidate) / 2
            vec = vec / np.linalg.norm(vec)
        # Match the historical order: sample instruction/blending before padding.
        if x.shape[1] < 2000:
            n = 2000 - x.shape[1]
            left = random.randint(0, n) if self.training else 0
            x = np.pad(x, ((0, 0), (left, n - left)))
        elif x.shape[1] > 2000:
            start = random.randint(0, x.shape[1] - 2000) if self.training else 0
            x = x[:, start:start + 2000]
        if y < 0:
            raise ValueError("Target held-out labels are isolated during preparation")
        return np.ascontiguousarray(x), np.asarray(vec, dtype=np.float32), self.key2idx[self.targets[ds][y]]


def load_context(config):
    np, torch, yaml, hist = runtime(config)
    prototypes, key2idx, text2emb = hist.build_prototype(hist.Targets, "mpnet-base")
    with gzip.open(config["index_path"], "rt") as f:
        index = json.load(f)
    return np, torch, yaml, hist, prototypes, key2idx, text2emb, index


def restrict_step(torch, optimizer, model, mask, baseline):
    """Enforce the same per-element invariant for either LAF optimizer."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            selected = mask[name]
            if p.grad is not None:
                p.grad.mul_(selected)
            for state in optimizer.state.get(p, {}).values():
                if torch.is_tensor(state) and state.shape == p.shape:
                    state.mul_(selected)
    optimizer.step()
    with torch.no_grad():
        for name, p in model.named_parameters():
            selected = mask[name]
            p.copy_(torch.where(selected, p, baseline[name]))
            for state in optimizer.state.get(p, {}).values():
                if torch.is_tensor(state) and state.shape == p.shape:
                    state.mul_(selected)
            if not torch.equal(p[~selected], baseline[name][~selected]):
                raise AssertionError(f"Unselected elements changed: {name}")


def make_vae(torch):
    nn = torch.nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
            self.mu, self.logvar = nn.Linear(64, 16), nn.Linear(64, 16)
            self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 768))

        def forward(self, x, sample=False):
            hidden = self.encoder(x)
            mu, lv = self.mu(hidden), self.logvar(hidden).clamp(-10, 10)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * lv) if sample else mu
            return self.decoder(z), mu, lv

    return VAE()


def selfcheck(config):
    _, torch, _, hist = runtime(config)
    hist.seed_all(42)
    model = torch.nn.Linear(7, 3)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    masks = {n: torch.arange(p.numel()).reshape(p.shape) % 2 == 0 for n, p in model.named_parameters()}
    opts = [torch.optim.Adam(model.parameters(), lr=lr) for lr in (1e-5, 4e-5)]
    for _ in range(3):
        for opt in opts:
            opt.zero_grad(set_to_none=True)
            model(torch.randn(5, 7)).square().mean().backward()
            restrict_step(torch, opt, model, masks, base)
    assert any(not torch.equal(p[masks[n]], base[n][masks[n]]) for n, p in model.named_parameters())
    vae = make_vae(torch).requires_grad_(False)
    x = torch.randn(3, 768, requires_grad=True)
    rec, _, _ = vae(x)
    a = (x - rec).square().sum(-1)
    torch.exp(a / (a + 1)).mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    write_json(Path(config["run_dir"]) / "checks/synthetic.json", {
        "masked_updates_and_both_optimizer_states": "passed", "unselected_exact": True,
        "vae_frozen_parameters_input_gradient": "passed", "real_data_used": False, "at": now()})


def preflight(config):
    np, torch, _, hist, proto, key2idx, text, index = load_context(config)
    artifact, run = Path(config["artifact_dir"]), Path(config["run_dir"])
    if digest(config["original_checkpoint"]) != config["original_sha256"]:
        raise ValueError("Original checkpoint hash changed")
    pre = config["input_checkpoint"]
    if digest(pre["path"]) != pre["sha256"]:
        raise ValueError("Pretrained Tower hash changed")
    raw = hist.load_yaml()
    module = hist.Instruct_Tuning_Module(hist.build_model_config(raw), hist.build_train_config(raw, "instruct"),
                                        proto, pre["path"], 653, seed=42)
    atomic_save(torch, module.model.state_dict(), config["initial_state"])
    state = torch.load(config["original_checkpoint"], map_location="cpu", weights_only=True)
    strict = module.model.load_state_dict(state, strict=True)
    model = module.model.to("cuda:0").eval()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = [x for x in index["train"] if x[0] == "MI_Shin2017A" and x[4] == "S01"]
    data = DataAccess(config, target, text, hist.Targets, hist.Instructions, key2idx)
    x, q, y = data[0]
    emb, _ = model(torch.from_numpy(x).unsqueeze(0).cuda(), torch.from_numpy(q).unsqueeze(0).cuda())
    assert tuple(emb.shape) == (1, 768) and torch.isfinite(emb).all()
    assert torch.allclose(emb.norm(dim=-1), torch.ones(1, device=emb.device), atol=1e-6)
    torch.nn.functional.cross_entropy(emb @ torch.tensor(proto.T, device=emb.device), torch.tensor([y], device=emb.device)).backward()
    active = {n: list(p.shape) for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("tower.") for n in active) and any(n.startswith("qformer.") for n in active)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for n, value in model.state_dict().items():
        assert torch.equal(value.cpu(), state[n]), n
    write_json(run / "checks/active-parameters.json", active)
    instructions = {}
    for ds, classes in hist.Targets.items():
        t = hist.Instructions[task_family(ds)][0]
        v = text[t] + np.mean([text[k] for k in classes], axis=0) / np.linalg.norm(np.mean([text[k] for k in classes], axis=0))
        v = np.asarray(v / np.linalg.norm(v), dtype=np.float32)
        instructions[ds] = {"text": t, "candidate_order": classes, "main_vector_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                            "default_text": "None", "supplementary": ["default_no_blend", "task_no_blend"]}
    write_json(run / "checks/instructions.json", instructions)
    write_json(run / "checks/preflight.json", {
        "at": now(), "strict_missing": list(strict.missing_keys), "strict_unexpected": list(strict.unexpected_keys),
        "state_keys": len(state), "embedding_shape": list(emb.shape), "unit_norm": emb.detach().norm().item(),
        "active_tensors": len(active), "active_elements": sum(p.numel() for n, p in model.named_parameters() if n in active),
        "inactive_parameters": [n for n, p in model.named_parameters() if p.grad is None],
        "initial_sha256": digest(config["initial_state"]), "initialization_limitation": config["initialization_note"],
        "parameters_and_buffers_unchanged": True, "target_heldout_and_test_arrays_loaded": False,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version},
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})


def retrain(config):
    np, torch, _, hist, proto, key2idx, text, index = load_context(config)
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    artifact = Path(config["artifact_dir"])
    out = Path(os.environ["LEAF_STAGE_ATTEMPT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    excluded = set(config["excluded_subjects"])
    def retain(row):
        return not (row[0] == "MI_Shin2017A" and row[4] in excluded)
    train, valid = [r for r in index["train"] if retain(r)], [r for r in index["valid"] if retain(r)]
    assert (len(train), len(valid)) == (len(index["train"])-48*len(excluded), len(index["valid"])-12*len(excluded))
    train_data = DataAccess(config, train, text, hist.Targets, hist.Instructions, key2idx, training=True, fixed=False)
    valid_data = DataAccess(config, valid, text, hist.Targets, hist.Instructions, key2idx, fixed=False)
    hist.seed_all(42)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    raw = hist.load_yaml()
    hist.fn_save_ckpt = lambda epoch: out / f"it{epoch+1:03d}.ckpt"

    class Module(hist.Instruct_Tuning_Module):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite instruction loss")
            if batch_idx % 50 == 0:
                write_json(artifact / "retrain-progress.json", {"at": now(), "epoch": self.current_epoch + 1,
                           "batch": batch_idx, "steps_per_epoch": len(train_loader), "global_step": self.global_step,
                           "loss": loss.detach().item(), "attempt": str(out),
                           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
                print(f"epoch={self.current_epoch+1} batch={batch_idx}/{len(train_loader)} loss={loss.detach().item():.6f}", flush=True)
            return loss

        def on_save_checkpoint(self, checkpoint):
            numpy_state = np.random.get_state()
            checkpoint["experiment_rng"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                                            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
            checkpoint["experiment_rng"]["numpy"] = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])

        def on_load_checkpoint(self, checkpoint):
            self.restore_rng = checkpoint.get("experiment_rng")

        def on_train_start(self):
            rng = getattr(self, "restore_rng", None)
            if rng:
                random.setstate(rng["python"])
                n = rng["numpy"]
                np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
                torch.set_rng_state(rng["cpu"].cpu())
                torch.cuda.set_rng_state_all([v.cpu() for v in rng["cuda"]])

        def on_validation_epoch_end(self):
            super().on_validation_epoch_end()
            if not self.trainer.sanity_checking:
                write_json(out / "validation.json", self.valLoss)

    module = Module(hist.build_model_config(raw), hist.build_train_config(raw, "instruct"),
                    proto, config["input_checkpoint"]["path"], len(train_loader), seed=42)
    module.model.load_state_dict(torch.load(config["initial_state"], map_location="cpu", weights_only=True), strict=True)
    checkpoint = ModelCheckpoint(dirpath=out / "resume", save_top_k=0, save_last=True,
                                 every_n_epochs=1, save_on_train_epoch_end=True)
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=100, precision="bf16-mixed",
                         benchmark=True, deterministic=False, enable_progress_bar=False,
                         enable_model_summary=False, logger=False, check_val_every_n_epoch=10,
                         callbacks=[checkpoint], num_sanity_val_steps=0)
    trainer.fit(module, train_loader, valid_loader, ckpt_path=os.environ.get("LEAF_RESUME_PATH") or None)
    final = artifact / "retrain_instruction/it100.ckpt"
    atomic_save(torch, module.model.state_dict(), final)
    write_json(artifact / "retrain_instruction/completed.json", {"at": now(), "checkpoint": str(final),
               "sha256": digest(final), "epochs": trainer.current_epoch, "global_step": trainer.global_step,
               "train_count": len(train), "valid_count": len(valid), "selected_epoch": 100,
               "target_valid_and_test_arrays_loaded": False, "source_attempt": str(out),
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20})


def importance(config):
    np, torch, _, hist, proto, key2idx, text, index = load_context(config)
    hist.seed_all(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    artifact, run = Path(config["artifact_dir"]), Path(config["run_dir"])
    folder = artifact / "importance"
    folder.mkdir(exist_ok=True)
    model = hist.LEAF(hist.build_model_config(hist.load_yaml()))
    original = torch.load(config["original_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(original, strict=True)
    model.cuda().eval()
    active = json.loads((run / "checks/active-parameters.json").read_text())
    parameters = {n: p for n, p in model.named_parameters() if n in active}
    data = DataAccess(config, index["train"], text, hist.Targets, hist.Instructions, key2idx)
    prototype = torch.tensor(proto.T, device="cuda:0")
    resume = folder / "recovery.pt"
    identity = {"model": config["original_sha256"], "index": digest(config["index_path"]),
                "instructions": digest(run / "checks/instructions.json"), "input_length": 2000,
                "loss": "33-class CE scale1 per-sample gradient square, eval FP32"}
    start, counts = 0, {s: 0 for s in ("overall", "S01", "S02", "S03")}
    if resume.exists():
        stored = torch.load(resume, map_location="cpu", weights_only=False)
        if stored["identity"] != identity:
            raise RuntimeError("Importance recovery identity mismatch")
        sums = {s: {n: t.cuda() for n, t in tensors.items()} for s, tensors in stored["sums"].items()}
        start, counts = stored["next"], stored["counts"]
    else:
        sums = {s: {n: torch.zeros_like(p) for n, p in parameters.items()} for s in counts}
    emb_path = folder / "original_train_embeddings.npy"
    if start and not emb_path.exists():
        raise RuntimeError("Importance recovery lacks paired embedding cache")
    features = np.lib.format.open_memmap(emb_path, mode="r+" if start else "w+", dtype="float32", shape=(len(data), 768))
    begin = time.monotonic()
    for i in range(start, len(data)):
        x, q, y = data[i]
        model.zero_grad(set_to_none=True)
        emb, _ = model(torch.from_numpy(x)[None].cuda(), torch.from_numpy(q)[None].cuda())
        loss = torch.nn.functional.cross_entropy(emb @ prototype, torch.tensor([y], device="cuda:0"))
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite importance loss at {i}")
        features[i] = emb.detach().cpu().numpy()[0]
        row = index["train"][i]
        subject = row[4] if row[0] == "MI_Shin2017A" and row[4] in counts else None
        with torch.no_grad():
            for n, p in parameters.items():
                if p.grad is None:
                    raise RuntimeError(f"Gradient disappeared: {n}")
                g2 = p.grad.square()
                sums["overall"][n].add_(g2)
                if subject:
                    sums[subject][n].add_(g2)
        counts["overall"] += 1
        if subject:
            counts[subject] += 1
        if (i + 1) % 250 == 0:
            write_json(folder / "progress.json", {"at": now(), "samples": i+1, "total": len(data),
                       "counts": counts, "elapsed_attempt_seconds": time.monotonic()-begin})
            print(f"importance {i+1}/{len(data)}", flush=True)
        if (i + 1) % 1000 == 0 or i + 1 == len(data):
            features.flush()
            assert all(torch.isfinite(t).all() and (t >= 0).all() for t in sums["overall"].values())
            atomic_save(torch, {"identity": identity, "next": i+1, "counts": counts,
                        "sums": {s: {n: t.cpu() for n, t in ts.items()} for s, ts in sums.items()}}, resume)
    assert counts == {"overall": 167091, "S01": 48, "S02": 48, "S03": 48}
    for n, value in model.state_dict().items():
        assert torch.equal(value.cpu(), original[n]), n
    means = {s: {n: t.cpu() / counts[s] for n, t in ts.items()} for s, ts in sums.items()}
    final = folder / "per-sample-importances.pt"
    atomic_save(torch, {"identity": identity, "counts": counts, "means": means}, final)
    mask = {n: means["S01"][n] > means["overall"][n] for n in parameters}
    generator = torch.Generator().manual_seed(42)
    random_mask, stats = {}, {}
    for n, selected in mask.items():
        k = int(selected.sum())
        other = torch.zeros(selected.numel(), dtype=torch.bool)
        other[torch.randperm(other.numel(), generator=generator)[:k]] = True
        random_mask[n] = other.reshape(selected.shape)
        stats[n] = {"selected": k, "total": selected.numel()}
    selected, total = sum(s["selected"] for s in stats.values()), sum(s["total"] for s in stats.values())
    if selected == 0 or selected == total:
        raise RuntimeError(f"Degenerate importance mask: {selected}/{total}")
    atomic_save(torch, {"importance": mask, "random": random_mask, "stats": stats,
                       "identity": identity, "seed": 42}, folder / "masks.pt")
    write_json(folder / "completed.json", {"at": now(), "identity": identity, "counts": counts,
               "importance_sha256": digest(final), "embedding_sha256": digest(emb_path),
               "masks_sha256": digest(folder / "masks.pt"), "selected": selected, "eligible": total,
               "selected_fraction": selected / total, "by_tensor": stats,
               "parameters_and_buffers_unchanged": True, "test_and_target_valid_arrays_loaded": False,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20})


def vae(config):
    np, torch, _, hist, _, _, _, index = load_context(config)
    artifact = Path(config["artifact_dir"])
    out = artifact / "vae"
    out.mkdir(exist_ok=True)
    features = np.load(artifact / "importance/original_train_embeddings.npy", mmap_mode="r")
    reports = {}
    for subject in ("overall", "S01", "S02", "S03"):
        hist.seed_all(42)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        selected = list(range(len(index["train"]))) if subject == "overall" else [i for i, row in enumerate(index["train"]) if row[0] == "MI_Shin2017A" and row[4] == subject]
        x = torch.from_numpy(np.array(features[selected], copy=True))
        loader = torch.utils.data.DataLoader(x, batch_size=128 if subject == "overall" else 16, shuffle=True)
        model = make_vae(torch).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        curve = []
        for epoch in range(20 if subject == "overall" else 50):
            total = 0.
            model.train()
            for batch in loader:
                batch = batch.cuda()
                opt.zero_grad(set_to_none=True)
                recon, mu, lv = model(batch, sample=True)
                error = (recon-batch).square().sum(-1).mean()
                kl = -.5 * (1 + lv - mu.square() - lv.exp()).sum(-1).mean()
                loss = error + .025 * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite VAE loss")
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            curve.append({"epoch": epoch+1, "loss": total/len(x)})
            print(f"VAE {subject} epoch={epoch+1} loss={total/len(x):.7f}", flush=True)
        model.eval().requires_grad_(False)
        errors, mus = [], []
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(x, batch_size=128):
                recon, mu, _ = model(batch.cuda())
                errors.append((recon-batch.cuda()).square().sum(-1).cpu())
                mus.append(mu.cpu())
        mse = torch.cat(errors)
        mu = torch.cat(mus)
        probe = x[:16].cuda().requires_grad_()
        recon, _, _ = model(probe)
        a = (probe-recon).square().sum(-1)
        torch.exp(a/(a+1)).mean().backward()
        assert torch.isfinite(probe.grad).all() and probe.grad.norm() > 0
        path = out / (subject + ".pt")
        atomic_save(torch, model.state_dict(), path)
        reports[subject] = {"n": len(x), "curve": curve, "checkpoint": str(path), "sha256": digest(path),
                            "deterministic_reconstruction_mse": mse.mean().item(),
                            "constant_mean_mse": (x-x.mean(0)).square().sum(-1).mean().item(),
                            "latent_mu_std": mu.std(0).mean().item(), "bounded_phi_mean": torch.exp(mse/(mse+1)).mean().item(),
                            "finite_nonzero_input_gradient_norm": probe.grad.norm().item()}
    write_json(out / "completed.json", {"at": now(), "models": reports, "gate": "requires_preparation_review_before_LAF",
                                        "auxiliary_caches": "retained locally on lab server; not published to GitHub"})


def supervise(config):
    """Select a single available GPU; OOM retries wait at least 20 minutes."""
    import yaml
    run, artifact = Path(config["run_dir"]), Path(config["artifact_dir"])
    lock = artifact / "supervisor.lock"
    import fcntl
    with lock.open("w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lf.write(str(os.getpid()))
        lf.flush()
        state_path = artifact / "runner-state.json"
        done = []
        stage_list = ["selfcheck", "preflight", "retrain", "importance", "vae"]
        for stage in stage_list:
            marker = artifact / "stage-results" / (stage + ".json")
            if marker.exists() and json.loads(marker.read_text())["exit_code"] == 0:
                done.append(stage)
                continue
            attempt = 0
            existing = list((artifact / "attempts" / stage).glob("attempt*"))
            if existing:
                attempt = max(int(p.name.replace("attempt", "")) for p in existing)
            minimum = 24000 if stage == "retrain" else 2500
            while True:
                inventory = gpu_inventory()
                gpu = max(inventory, key=lambda v: v["free_mib"])
                if stage != "selfcheck" and gpu["free_mib"] < minimum:
                    write_json(state_path, {"status": "waiting_gpu_memory", "stage": stage, "at": now(),
                               "minimum_free_mib": minimum, "inventory": inventory, "completed_stages": done})
                    time.sleep(60)
                    continue
                attempt += 1
                directory = artifact / "attempts" / stage / f"attempt{attempt:03d}"
                directory.mkdir(parents=True)
                stage_config = yaml.safe_load((run / "configs" / (stage + ".yaml")).read_text())
                stage_config["attempt"] = attempt
                stage_config["execution"].update(physical_gpu=gpu["index"], gpu_uuid=gpu["uuid"],
                                                    CUDA_VISIBLE_DEVICES=gpu["uuid"], attempt_output=str(directory))
                resumes = sorted((artifact / "attempts/retrain").glob("attempt*/resume/last.ckpt"), key=lambda p:p.stat().st_mtime)
                resume = str(resumes[-1]) if stage == "retrain" and resumes else ""
                stage_config["execution"]["resume_checkpoint"] = resume or None
                stage_config["execution"]["script_sha256"] = digest(artifact / "run_instruction_unlearning.py")
                if Path(config["initial_state"]).exists():
                    stage_config["initial_state_sha256"] = digest(config["initial_state"])
                conf_dir = run / "configs/attempts" / stage
                conf_dir.mkdir(parents=True, exist_ok=True)
                conf_path = conf_dir / f"attempt{attempt:03d}.yaml"
                conf_path.write_text(yaml.safe_dump(stage_config, allow_unicode=True, sort_keys=False))
                manifest_path = run / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest.setdefault("attempt_configs", []).append({"path": str(conf_path), "sha256": digest(conf_path)})
                manifest.update(status="running_preparation", current_stage=stage, updated_at=now())
                write_json(manifest_path, manifest)
                log_path = run / "logs" / f"{stage}-attempt{attempt:03d}.log"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                           PYTHONUNBUFFERED="1", LEAF_STAGE_ATTEMPT_DIR=str(directory), LEAF_RESUME_PATH=resume)
                env["LEAF_EFFECTIVE_CONFIG"] = str(conf_path)
                cmd = [sys.executable, "-u", str(artifact / "run_instruction_unlearning.py"), "--run-dir", str(run), stage]
                started_at, started_clock = now(), time.monotonic()
                with log_path.open("w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                    write_json(state_path, {"status": "running", "stage": stage, "attempt": attempt, "child_pid": proc.pid,
                               "supervisor_pid": os.getpid(), "gpu": gpu, "started_at": started_at, "log": str(log_path),
                               "config": str(conf_path), "config_sha256": digest(conf_path), "completed_stages": done})
                    code = proc.wait()
                result = {"stage": stage, "attempt": attempt, "exit_code": code, "finished_at": now(),
                          "started_at": started_at, "elapsed_seconds": time.monotonic() - started_clock,
                          "log": str(log_path), "config": str(conf_path), "config_sha256": digest(conf_path), "gpu": gpu}
                write_json(directory / "result.json", result)
                (run / "logs" / f"{stage}-attempt{attempt:03d}.exitcode").write_text(str(code)+"\n")
                if code == 0:
                    write_json(marker, result)
                    done.append(stage)
                    break
                with log_path.open("rb") as f:
                    f.seek(max(0, f.seek(0, 2)-65536))
                    tail = f.read().decode(errors="replace").lower()
                oom = "out of memory" in tail or "cuda error: memory allocation" in tail
                if not oom:
                    write_json(state_path, dict(result, status="failed_requires_diagnosis", completed_stages=done))
                    return 1
                minimum = min(gpu["total_mib"] - 1024, max(minimum, gpu["free_mib"] + 2048))
                write_json(state_path, dict(result, status="oom_cooldown", cooldown_seconds=1200,
                           retry_not_before_unix=time.time()+1200, next_minimum_free_mib=minimum, completed_stages=done))
                time.sleep(1200)
        write_json(state_path, {"status": "preparation_complete_requires_review", "at": now(), "completed_stages": done,
                               "next": "Review masks/VAE checks, implement and run all predetermined LAF comparisons; then synchronized evaluation, continued learning and sequential requests."})
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", default="/userdata/lwk/EEG&LLM/Projects/LEAF-main")
    parser.add_argument("stage", choices=["initialize", "selfcheck", "preflight", "retrain", "importance", "vae", "supervise"])
    args = parser.parse_args()
    if args.stage == "initialize":
        initialize(args)
        return 0
    import yaml
    stage = "prepare" if args.stage == "supervise" else args.stage
    path = Path(os.environ.get("LEAF_EFFECTIVE_CONFIG", str(Path(args.run_dir) / "configs" / (stage + ".yaml"))))
    config = yaml.safe_load(path.read_text())
    return globals()[args.stage](config) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
