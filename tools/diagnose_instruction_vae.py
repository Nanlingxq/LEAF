"""Training-only, separately archived VAE budget diagnosis; not a main result."""

import argparse
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def load_preparation(path):
    spec = importlib.util.spec_from_file_location("instruction_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(config, prep):
    np, torch, _, hist = prep.runtime(config)
    out = Path(config["diagnostic_output"])
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(config["index_path"], "rt") as f:
        index = json.load(f)
    features_path = Path(config["artifact_dir"]) / "importance/original_train_embeddings.npy"
    assert prep.digest(features_path) == config["diagnostic_inputs"]["embeddings_sha256"]
    features = np.load(features_path, mmap_mode="r")
    reports = {}
    for subject in ("S01", "S02", "S03"):
        hist.seed_all(42)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = [i for i, row in enumerate(index["train"]) if row[0] == "MI_Shin2017A" and row[4] == subject]
        assert len(rows) == 48
        x = torch.from_numpy(np.array(features[rows], copy=True))
        loader = torch.utils.data.DataLoader(x, batch_size=16, shuffle=True)
        model = prep.make_vae(torch).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        curve, probes = [], {}
        for epoch in range(200):
            model.train()
            loss_sum = 0.
            for batch in loader:
                batch = batch.cuda()
                optimizer.zero_grad(set_to_none=True)
                recon, mu, lv = model(batch, sample=True)
                mse = (recon - batch).square().sum(-1).mean()
                kl = -.5 * (1 + lv - mu.square() - lv.exp()).sum(-1).mean()
                loss = mse + .025 * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite diagnostic VAE loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(batch)
            curve.append({"epoch": epoch+1, "loss": loss_sum/48})
            if epoch+1 in (50, 100, 200):
                model.eval()
                with torch.no_grad():
                    rec, mu, _ = model(x.cuda())
                    error = (rec-x.cuda()).square().sum(-1).mean().item()
                    baseline = (x-x.mean(0)).square().sum(-1).mean().item()
                    shifted = model.decoder(mu.roll(1, 0))
                    penalty = ((shifted-x.cuda()).square().sum(-1).mean()).item()-error
                checkpoint = out / f"{subject}-epoch{epoch+1:03d}.pt"
                prep.atomic_save(torch, model.state_dict(), checkpoint)
                probe = {"deterministic_mse": error, "constant_mean_mse": baseline,
                         "mse_ratio": error/baseline, "latent_mu_std": mu.std(0).mean().item(),
                         "cyclic_permutation_mse_penalty": penalty,
                         "checkpoint": str(checkpoint), "sha256": prep.digest(checkpoint)}
                if epoch+1 == 50:
                    prior = torch.load(config["diagnostic_inputs"]["original_target_vaes"][subject]["checkpoint"],
                                       map_location="cpu", weights_only=True)
                    probe["matches_original_50epoch_tensors"] = all(torch.equal(t.cpu(), prior[k]) for k, t in model.state_dict().items())
                    probe["max_tensor_difference_to_original_50epoch"] = max((t.cpu()-prior[k]).abs().max().item() for k, t in model.state_dict().items())
                probes[str(epoch+1)] = probe
                print(subject, epoch+1, json.dumps(probe), flush=True)
        model.eval().requires_grad_(False)
        sample = x[:16].cuda().requires_grad_()
        rec, _, _ = model(sample)
        mse = (sample-rec).square().sum(-1)
        torch.exp(mse/(mse+1)).mean().backward()
        assert torch.isfinite(sample.grad).all() and sample.grad.norm() > 0
        reports[subject] = {"n": 48, "optimization_steps": 600, "curve": curve, "probes": probes,
                            "input_gradient_norm": sample.grad.norm().item()}
        prep.write_json(out / "progress.json", {"at": prep.now(), "completed_subjects": list(reports)})
    prep.write_json(out / "completed.json", {"at": prep.now(), "purpose": "separate training-only budget diagnosis",
                    "main_plan_budget_changed": False, "target_valid_or_test_used": False,
                    "models": reports, "primary_use": "requires_explicit_protocol_decision"})


def supervise(config, prep, config_path):
    out = Path(config["diagnostic_output"])
    out.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        gpu = max(prep.gpu_inventory(), key=lambda x: x["free_mib"])
        if gpu["free_mib"] < 2500:
            prep.write_json(out / "runner-state.json", {"status": "waiting_gpu", "gpu": gpu, "at": prep.now()})
            time.sleep(60)
            continue
        attempt += 1
        effective = copy.deepcopy(config)
        effective["attempt"] = attempt
        effective["execution"].update(physical_gpu=gpu["index"], gpu_uuid=gpu["uuid"], CUDA_VISIBLE_DEVICES=gpu["uuid"])
        effective_path = config_path.with_name(config_path.stem + f"-attempt{attempt:03d}.yaml")
        import yaml
        effective_path.write_text(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True))
        run = Path(config["run_dir"])
        manifest = json.loads((run/"manifest.json").read_text())
        manifest.setdefault("diagnostic_configs", []).append({"path": str(effective_path), "sha256": prep.digest(effective_path)})
        prep.write_json(run/"manifest.json", manifest)
        log_path = out / f"attempt{attempt:03d}.log"
        with log_path.open("w") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(effective_path), "--worker"]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], OMP_NUM_THREADS="8"))
            prep.write_json(out/"runner-state.json", {"status": "running", "pid": process.pid,
                            "at": prep.now(), "gpu": gpu, "config": str(effective_path), "log": str(log_path)})
            code = process.wait()
        (out/f"attempt{attempt:03d}.exitcode").write_text(str(code)+"\n")
        if code == 0:
            prep.write_json(out/"runner-state.json", {"status": "completed_diagnostic_only", "at": prep.now(), "exit_code": 0})
            return 0
        text = log_path.read_text(errors="replace").lower()
        if "out of memory" not in text:
            prep.write_json(out/"runner-state.json", {"status": "failed", "at": prep.now(), "exit_code": code, "log": str(log_path)})
            return code
        prep.write_json(out/"runner-state.json", {"status": "oom_cooldown", "retry_not_before_unix": time.time()+1200, "at": prep.now()})
        time.sleep(1200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    prep = load_preparation(config["preparation_script"])
    if args.worker:
        worker(config, prep)
        return 0
    return supervise(config, prep, args.config)


if __name__ == "__main__":
    sys.exit(main())
