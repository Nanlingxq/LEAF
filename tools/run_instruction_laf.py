"""Label-free immediate SSD/LAF comparisons after audited preparation.

Requests consume a label-free training manifest, fixed teacher embeddings,
importance statistics and VAEs. They never load validation or test arrays.
"""

import argparse
import copy
import csv
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


VARIANTS = ["ssd_only", "laf_all", "laf_random_mask", "laf_importance_mask",
            "ssd_then_laf_mask", "ue_only", "ra_only"]


def preparation(config):
    spec = importlib.util.spec_from_file_location("instruction_preparation", config["preparation_script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ue_loss(torch, retained, target, full_vae, target_vae):
    full, _, _ = full_vae(retained)
    forgotten, _, _ = target_vae(target)
    a = (retained-full).square().sum(-1)
    b = (target-forgotten).square().sum(-1)
    return torch.exp(a/(a+1)).mean()-torch.exp(b/(b+1)).mean()


def ra_loss(torch, retained, target, retained_teacher, target_teacher, tau=20.):
    r = 1-torch.nn.functional.cosine_similarity(retained, retained_teacher)
    f = 1-torch.nn.functional.cosine_similarity(target, target_teacher)
    return r.mean()-(torch.logsumexp(f/tau, dim=0)-math.log(len(f)))


def dampen(torch, model, masks, overall, target):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in masks:
                continue
            selected = masks[name].to(p.device)
            if selected.any():
                a, b = overall[name].to(p.device), target[name].to(p.device)
                assert (b[selected] > 0).all()
                scale = (a[selected]/b[selected]).clamp(max=1)
                p[selected] *= scale


class UnlabeledInputs:
    def __init__(self, config, records, vectors):
        self.records, self.vectors = records, vectors
        self.paths = {s["dataset"]: s["path"] for s in config["datasets"]}
        self.files = {}

    def batch(self, indices, np, torch):
        import h5py
        xs, qs = [], []
        for i in indices:
            row = self.records[i]
            ds = row["dataset"]
            if ds not in self.files:
                self.files[ds] = h5py.File(self.paths[ds], "r", locking=False)
            x = np.asarray(self.files[ds][row["x_path"]][row["trial"]], dtype=np.float32)
            if not np.isfinite(x).all():
                raise FloatingPointError(f"Non-finite training input {i}")
            if x.shape[1] < 2000:
                x = np.pad(x, ((0, 0), (0, 2000-x.shape[1])))
            else:
                x = x[:, :2000]
            xs.append(x)
            qs.append(self.vectors[ds])
        return torch.from_numpy(np.stack(xs)).cuda(), torch.from_numpy(np.stack(qs)).cuda()


def synthetic(config, prep):
    _, torch, _, hist = prep.runtime(config)
    hist.seed_all(42)
    full, target = prep.make_vae(torch).requires_grad_(False), prep.make_vae(torch).requires_grad_(False)
    r, f = torch.randn(3, 768, requires_grad=True), torch.randn(3, 768, requires_grad=True)
    loss = ue_loss(torch, r, f, full, target)
    loss.backward()
    assert torch.isfinite(r.grad).all() and torch.isfinite(f.grad).all()
    assert r.grad.norm() > 0 and f.grad.norm() > 0
    assert all(p.grad is None for v in (full, target) for p in v.parameters())
    x = torch.randn(4, 768, requires_grad=True)
    ra_loss(torch, x, x, x.detach(), x.detach()).backward()
    assert x.grad.abs().max() < 1e-6
    toy = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad(): toy.weight.fill_(2.)
    mask = {"weight": torch.tensor([[True, False, True], [False, True, False]])}
    dampen(torch, toy, mask, {"weight": torch.ones(2, 3)}, {"weight": torch.full((2, 3), 4.)})
    assert torch.equal(toy.weight[mask["weight"]], torch.full((3,), .5))
    assert torch.equal(toy.weight[~mask["weight"]], torch.full((3,), 2.))
    prep.write_json(Path(config["run_dir"])/"checks/laf-synthetic.json", {
        "at": prep.now(), "UE_frozen_VAE_input_gradient": "passed", "RA_identical_embedding_gradient": "near_zero",
        "SSD_ratio_and_unselected_elements": "passed", "real_data_used": False})


def worker(config, variant, prep):
    np, torch, _, hist = prep.runtime(config)
    hist.seed_all(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    artifact, run = Path(config["artifact_dir"]), Path(config["run_dir"])
    out = Path(os.environ["LEAF_LAF_ATTEMPT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    assert prep.digest(config["unlabeled_manifest"]) == config["unlabeled_manifest_sha256"]
    assert prep.digest(config["paired_plan"]) == config["paired_plan_sha256"]
    assert prep.digest(config["original_checkpoint"]) == config["original_sha256"]
    assert prep.digest(config["masks_path"]) == config["masks_sha256"]
    with gzip.open(config["unlabeled_manifest"], "rt") as f:
        records = json.load(f)
    # The request manifest has no per-trial label field or original labeled rows.
    assert all(set(row)=={"dataset", "x_path", "trial", "subject", "cache_row"} for row in records)
    pairs = json.loads(Path(config["paired_plan"]).read_text())
    text = hist.load_embeddings("mpnet-base")
    fixed = json.loads((run/"checks/instructions.json").read_text())
    vectors = {}
    for ds, classes in hist.Targets.items():
        candidate = np.mean([text[t] for t in classes], axis=0)
        candidate /= np.linalg.norm(candidate)
        vector = (text[hist.Instructions[prep.task_family(ds)][0]]+candidate)/2
        vector = np.asarray(vector/np.linalg.norm(vector), dtype=np.float32)
        import hashlib
        assert hashlib.sha256(vector.tobytes()).hexdigest()==fixed[ds]["main_vector_sha256"]
        vectors[ds] = vector
    data = UnlabeledInputs(config, records, vectors)
    teacher_path = Path(config["teacher_embeddings"])
    assert prep.digest(teacher_path)==config["teacher_embeddings_sha256"]
    teacher = np.load(teacher_path, mmap_mode="r")
    model = hist.LEAF(hist.build_model_config(hist.load_yaml()))
    original = torch.load(config["original_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(original, strict=True)
    model.cuda().eval().requires_grad_(False)
    active = json.loads((run/"checks/active-parameters.json").read_text())
    saved_masks = torch.load(config["masks_path"], map_location="cpu", weights_only=True)
    mode = "random" if variant=="laf_random_mask" else "importance"
    masks = {n: (torch.ones_like(p, dtype=torch.bool) if variant=="laf_all" and n in active
                 else saved_masks[mode][n].cuda() if n in active else torch.zeros_like(p, dtype=torch.bool))
             for n, p in model.named_parameters()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    damp_seconds = 0.
    if variant in ("ssd_only", "ssd_then_laf_mask"):
        statistics = torch.load(config["importance_path"], map_location="cpu", weights_only=True)
        torch.cuda.synchronize()
        damp_started = time.monotonic()
        dampen(torch, model, masks, statistics["means"]["overall"], statistics["means"]["S01"])
        torch.cuda.synchronize()
        damp_seconds = time.monotonic()-damp_started
        del statistics
    baseline = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: t.detach().clone() for n, t in model.named_buffers()}
    for n, p in model.named_parameters():
        p.requires_grad_(n in active and variant!="ssd_only")
    prep.atomic_save(torch, model.state_dict(), out/"epoch000.ckpt")
    torch.cuda.synchronize()
    started = time.monotonic()
    curve = []
    if variant!="ssd_only":
        vaes = []
        for spec in (config["full_vae"], config["target_vae"]):
            assert prep.digest(spec["path"])==spec["sha256"]
            v = prep.make_vae(torch).cuda().eval().requires_grad_(False)
            v.load_state_dict(torch.load(spec["path"], map_location="cpu", weights_only=True), strict=True)
            vaes.append(v)
        full_vae, target_vae = vaes
        params = [p for p in model.parameters() if p.requires_grad]
        ue = torch.optim.Adam(params, lr=1e-5, betas=(.9,.999), eps=1e-8, weight_decay=0)
        ra = torch.optim.Adam(params, lr=4e-5, betas=(.9,.999), eps=1e-8, weight_decay=0)
        use_ue, use_ra = variant!="ra_only", variant!="ue_only"
        if variant=="real_check":
            pairs = [{"epoch": 1, "target": pairs[0]["target"][:2], "retained": pairs[0]["retained"][:2]}]
        for step, pair in enumerate(pairs):
            target_ids, retained_ids = pair["target"], pair["retained"]
            size = len(retained_ids)
            x, q = data.batch(retained_ids+target_ids, np, torch)
            record = {"epoch": pair["epoch"], "paired_batch": step+1, "retained_count": size, "target_count": len(target_ids)}
            for label, optimizer in (("UE", ue), ("RA", ra)):
                if (label=="UE" and not use_ue) or (label=="RA" and not use_ra):
                    continue
                optimizer.zero_grad(set_to_none=True)
                embeddings, _ = model(x, q)
                er, ef = embeddings[:size], embeddings[size:]
                if label=="UE":
                    loss = ue_loss(torch, er, ef, full_vae, target_vae)
                else:
                    tr = torch.from_numpy(np.array(teacher[retained_ids], copy=True)).cuda()
                    tf = torch.from_numpy(np.array(teacher[target_ids], copy=True)).cuda()
                    loss = ra_loss(torch, er, ef, tr, tf, tau=20.)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {label} loss")
                loss.backward()
                norm_squared = torch.zeros((), device="cuda:0")
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise FloatingPointError(f"Non-finite gradient: {name}")
                        norm_squared += (p.grad*masks[name]).square().sum()
                record[label+"_loss"] = loss.detach().item()
                record[label+"_selected_gradient_norm"] = norm_squared.sqrt().item()
                prep.restrict_step(torch, optimizer, model, masks, baseline)
                for name, value in model.named_buffers():
                    assert torch.equal(value, buffers[name]), name
            curve.append(record)
            prep.write_json(out/"curve.json", curve)
            prep.write_json(artifact/"immediate/progress.json", {"at": prep.now(), "variant": variant,
                            "epoch": pair["epoch"], "paired_batch": step+1, "paired_batches": len(pairs), "record": record})
            print(variant, json.dumps(record), flush=True)
            if (step+1)%2==0 and pair["epoch"]%10==0:
                prep.atomic_save(torch, model.state_dict(), out/f"epoch{pair['epoch']:03d}.ckpt")
        assert all(p.grad is None for v in vaes for p in v.parameters())
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    changed, by_tensor = 0, {}
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name
        original_p = original[name].to(p.device)
        different = p.detach()!=original_p
        n = int(different.sum())
        changed += n
        by_tensor[name] = {"changed": n, "selected": int(masks[name].sum()), "total": p.numel()}
        if name not in active:
            assert torch.equal(p, original_p), name
        assert torch.equal(p.detach()[~masks[name]], original_p[~masks[name]]), name
    for name, value in model.named_buffers():
        assert torch.equal(value, original[name].to(value.device)), name
    if variant=="real_check":
        assert changed>0 and any(row.get("UE_selected_gradient_norm", 0)>0 for row in curve)
    final = out/("real-check.ckpt" if variant=="real_check" else "final.ckpt")
    prep.atomic_save(torch, model.state_dict(), final)
    steps = 0 if variant=="ssd_only" else len(pairs)*(int(use_ue)+int(use_ra))
    report = {"variant": variant, "at": prep.now(), "checkpoint": str(final), "sha256": prep.digest(final),
              "epochs": 0 if variant=="ssd_only" else 1 if variant=="real_check" else 40,
              "optimizer_steps": steps, "changed_elements": changed, "by_tensor": by_tensor,
              "unselected_exactly_unchanged": True, "inactive_parameters_unchanged": True, "buffers_unchanged": True,
              "text_cache_frozen": True, "teacher_source": "fixed original embeddings cache", "request_task_labels_accessed": False,
              "target_validation_or_tests_loaded": False, "target_VAE_epochs": 200,
              "elapsed_gpu_synchronized_seconds": elapsed+damp_seconds,
              "direct_damp_gpu_seconds": damp_seconds,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
    prep.write_json(out/"completed.json", report)


def supervise(config, prep):
    import fcntl
    import yaml
    artifact, run = Path(config["artifact_dir"]), Path(config["run_dir"])
    folder = artifact/"immediate"
    folder.mkdir(exist_ok=True)
    state_path = folder/"runner-state.json"
    with (folder/"supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid()));lock.flush()
        results = {}
        for variant in ["real_check"]+VARIANTS:
            marker = folder/"results"/(variant+".json")
            if marker.exists():
                results[variant] = json.loads(marker.read_text())
                continue
            attempt = len(list((folder/"attempts"/variant).glob("attempt*")))
            minimum = 3000 if variant in ("real_check", "ssd_only") else 16000
            while True:
                gpu = max(prep.gpu_inventory(), key=lambda v:v["free_mib"])
                if gpu["free_mib"]<minimum:
                    prep.write_json(state_path, {"status":"waiting_gpu_memory", "variant":variant, "at":prep.now(),
                                                "gpu":gpu, "minimum_free_mib":minimum})
                    time.sleep(60)
                    continue
                attempt += 1
                out = folder/"attempts"/variant/f"attempt{attempt:03d}"
                out.mkdir(parents=True)
                effective = copy.deepcopy(config)
                effective.update(variant=variant, attempt=attempt)
                effective["execution"].update(physical_gpu=gpu["index"], gpu_uuid=gpu["uuid"], CUDA_VISIBLE_DEVICES=gpu["uuid"], attempt_output=str(out))
                conf_dir = run/"configs/immediate"/variant
                conf_dir.mkdir(parents=True,exist_ok=True)
                conf = conf_dir/f"attempt{attempt:03d}.yaml"
                effective["execution"]["command"] = [sys.executable,"-u",str(Path(__file__).resolve()),"--config",str(conf),"--worker",variant]
                conf.write_text(yaml.safe_dump(effective,sort_keys=False,allow_unicode=True))
                manifest_path=run/"manifest.json";manifest=json.loads(manifest_path.read_text())
                manifest.setdefault("immediate_configs",[]).append({"path":str(conf),"sha256":prep.digest(conf)})
                prep.write_json(manifest_path,manifest)
                log_path=run/"logs"/f"laf-{variant}-attempt{attempt:03d}.log"
                started=prep.now()
                with log_path.open("w") as log:
                    child=subprocess.Popen(effective["execution"]["command"],stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu["uuid"],OMP_NUM_THREADS="8",LEAF_LAF_ATTEMPT_DIR=str(out)))
                    prep.write_json(state_path,{"status":"running","variant":variant,"attempt":attempt,"child_pid":child.pid,
                        "supervisor_pid":os.getpid(),"at":started,"gpu":gpu,"log":str(log_path),"config":str(conf),"completed_variants":list(results)})
                    code=child.wait()
                (run/"logs"/f"laf-{variant}-attempt{attempt:03d}.exitcode").write_text(str(code)+"\n")
                if code==0:
                    result=json.loads((out/"completed.json").read_text())
                    result.update(exit_code=0,config=str(conf),config_sha256=prep.digest(conf),started_at=started)
                    prep.write_json(marker,result);results[variant]=result
                    break
                text=log_path.read_text(errors="replace").lower()
                if "out of memory" not in text:
                    prep.write_json(state_path,{"status":"failed_requires_diagnosis","variant":variant,"exit_code":code,"at":prep.now(),"log":str(log_path)})
                    return code
                minimum=min(gpu["total_mib"]-1024,max(minimum,gpu["free_mib"]+2048))
                prep.write_json(state_path,{"status":"oom_cooldown","variant":variant,"at":prep.now(),
                    "retry_not_before_unix":time.time()+1200,"next_minimum_free_mib":minimum,"log":str(log_path)})
                time.sleep(1200)
        bank={"original":{"checkpoint":config["original_checkpoint"],"sha256":config["original_sha256"]},
              "retrain_instruction":config["retrain_reference"]}
        for variant in VARIANTS:
            result=results[variant];bank[variant]={"checkpoint":result["checkpoint"],"sha256":result["sha256"]}
        assert all(prep.digest(v["checkpoint"])==v["sha256"] for v in bank.values())
        prep.write_json(folder/"candidate-bank-sealed.json",{"at":prep.now(),"models":bank,"target_validation_or_tests_loaded":False,
                        "target_VAE_epochs":200,"next":"Implement and execute unified evaluations only after this seal; then continued learning and sequential requests."})
        prep.write_json(state_path,{"status":"all_immediate_weights_fixed_evaluation_pending","at":prep.now(),"completed_variants":list(results)})
        return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--worker",choices=VARIANTS+["real_check"])
    parser.add_argument("--synthetic",action="store_true")
    args=parser.parse_args()
    import yaml
    config=yaml.safe_load(args.config.read_text());prep=preparation(config)
    if args.synthetic:
        synthetic(config,prep);return 0
    if args.worker:
        worker(config,args.worker,prep);return 0
    return supervise(config,prep)


if __name__=="__main__":
    sys.exit(main())
