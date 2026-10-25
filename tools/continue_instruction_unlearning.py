"""Fixed-budget retained-data learning after sealed unlearning comparisons."""
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

MODELS = ['retrain_instruction', 'original', 'ssd_only', 'laf_all', 'laf_importance_mask', 'ssd_then_laf_mask']
CONDITIONS = ['all_retained', 'non_mi']

def module(path):
    spec=importlib.util.spec_from_file_location('continued_preparation',path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def sample_plan(rows, tasks, np):
    rng=np.random.default_rng(42)
    pools={ds:[i for i,row in enumerate(rows) if row[0]==ds and not(row[0]=='MI_Shin2017A' and row[4]=='S01')] for ds in tasks}
    assert all(pools.values())
    return [[int(rng.choice(pools[tasks[(step*32+j)%len(tasks)]])) for j in range(32)] for step in range(200)]

def prepare(c,p):
    import numpy as np
    with gzip.open(c['index_path'],'rt') as f:rows=json.load(f)['train']
    tasks=[s['dataset'] for s in c['datasets']]
    plans={key:sample_plan(rows,[ds for ds in tasks if key=='all_retained' or not ds.startswith('MI_')],np) for key in CONDITIONS}
    assert len(tasks)==17 and len([ds for ds in tasks if not ds.startswith('MI_')])==9
    for key,steps in plans.items():
        assert len(steps)==200 and all(len(batch)==32 for batch in steps)
        assert all(not(rows[i][0]=='MI_Shin2017A' and rows[i][4]=='S01') for batch in steps for i in batch)
        if key=='non_mi':assert all(not rows[i][0].startswith('MI_') for batch in steps for i in batch)
    p.write_json(c['continued_plan'],{'seed':42,'index_sha256':p.digest(c['index_path']),'plans':plans,'target_excluded':True,'label_used_for_sampling':False})
    p.write_json(Path(c['run_dir'])/'checks/continued-data-ready.json',{'plan_sha256':p.digest(c['continued_plan']),'conditions':CONDITIONS,'models':MODELS,'checkpoints':[0,50,100,200],'all_sampled_trials_are_historical_retained_training':True})

def worker(c,p):
    np,torch,_,hist,proto,key2idx,text,index=p.load_context(c)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    hist.seed_all(42);torch.cuda.set_device(0)
    assert p.digest(c['continued_plan'])==c['continued_plan_sha256']
    plan=json.loads(Path(c['continued_plan']).read_text())
    assert plan['index_sha256']==p.digest(c['index_path'])
    spec=c['models'][c['identity']];assert p.digest(spec['checkpoint'])==spec['sha256']
    model=hist.LEAF(hist.build_model_config(hist.load_yaml()))
    model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model.cuda().train()
    active=json.loads((Path(c['run_dir'])/'checks/active-parameters.json').read_text())
    # The preparation file identifies the tensors on the instruction forward path.
    active_names=set(active['active'] if 'active' in active else active)
    for name,param in model.named_parameters():param.requires_grad_(name in active_names)
    assert sum(p.requires_grad for p in model.parameters())==383
    baseline={name:v.detach().cpu().clone() for name,v in model.named_buffers()}
    frozen={name:v.detach().cpu().clone() for name,v in model.named_parameters() if not v.requires_grad}
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-3)
    prototypes=torch.as_tensor(proto,dtype=torch.float32,device='cuda').detach()
    data=p.DataAccess(c,index['train'],text,hist.Targets,hist.Instructions,key2idx,training=False,fixed=True)
    out=Path(os.environ['LEAF_CONTINUED_ATTEMPT_DIR']);out.mkdir(parents=True,exist_ok=True)
    saved={}
    def save(step):
        path=out/f'step{step:03d}.ckpt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        saved[str(step)]={'checkpoint':str(path),'sha256':p.digest(path)}
    save(0);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic();curve=[]
    for step,ids in enumerate(plan['plans'][c['condition']],1):
        items=[data[i] for i in ids]
        x=torch.from_numpy(np.stack([v[0] for v in items])).cuda();q=torch.from_numpy(np.stack([v[1] for v in items])).cuda()
        y=torch.tensor([v[2] for v in items],dtype=torch.long,device='cuda')
        optimizer.zero_grad(set_to_none=True);emb,_=model(x,q)
        loss=torch.nn.functional.cross_entropy(emb@prototypes.T,y)
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        optimizer.step();curve.append({'step':step,'loss':float(loss.detach())})
        if step in (50,100,200):save(step)
        if step%10==0:
            p.write_json(Path(c['artifact_dir'])/'continued/progress.json',{'at':p.now(),'model':c['identity'],'condition':c['condition'],'step':step,'loss':curve[-1]['loss']})
            print(c['condition'],c['identity'],step,curve[-1]['loss'],flush=True)
    torch.cuda.synchronize()
    assert all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in model.named_parameters() if k in frozen)
    buffers={k:{'equal_to_start':torch.equal(v.detach().cpu(),baseline[k]),'changed_elements':int((v.detach().cpu()!=baseline[k]).sum())} for k,v in model.named_buffers()}
    p.write_json(out/'losses.json',curve)
    p.write_json(out/'completed.json',{'at':p.now(),'model':c['identity'],'condition':c['condition'],'steps':200,'checkpoints':saved,'seconds':time.monotonic()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'buffers':buffers,'inactive_parameters_unchanged':True,'evaluation_pending':True})

def supervise(c,p):
    import fcntl,yaml
    out=Path(c['artifact_dir'])/'continued';out.mkdir(exist_ok=True);run=Path(c['run_dir'])
    with (out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for condition in CONDITIONS:
            for identity in MODELS:
                task=condition+'--'+identity;marker=out/'results'/(task+'.json')
                if marker.exists():continue
                count=len(list((out/'attempts'/task).glob('attempt*')));minimum=14000
                while True:
                    gpu=max(p.gpu_inventory(),key=lambda g:g['free_mib'])
                    if gpu['free_mib']<minimum:
                        p.write_json(out/'runner-state.json',{'status':'waiting_gpu','task':task,'gpu':gpu,'at':p.now()});time.sleep(60);continue
                    count+=1;folder=out/'attempts'/task/f'attempt{count:03d}';folder.mkdir(parents=True)
                    cfg=copy.deepcopy(c);cfg.update(condition=condition,identity=identity,attempt=count)
                    path=run/'configs/continued'/task/f'attempt{count:03d}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(path),'--worker']
                    cfg['execution'].update(command=command,physical_gpu=gpu['index'],gpu_uuid=gpu['uuid'],CUDA_VISIBLE_DEVICES=gpu['uuid'])
                    path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
                    mp=run/'manifest.json';m=json.loads(mp.read_text());m.setdefault('continued_configs',[]).append({'path':str(path),'sha256':p.digest(path)});p.write_json(mp,m)
                    log=run/'logs'/f'continued-{task}-attempt{count:03d}.log'
                    with log.open('w') as f:
                        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],LEAF_CONTINUED_ATTEMPT_DIR=str(folder),OMP_NUM_THREADS='8'))
                        p.write_json(out/'runner-state.json',{'status':'running','task':task,'child_pid':child.pid,'supervisor_pid':os.getpid(),'gpu':gpu,'log':str(log),'config':str(path),'at':p.now()})
                        code=child.wait()
                    log.with_suffix('.exitcode').write_text(str(code)+'\n')
                    if code==0:
                        p.write_json(marker,json.loads((folder/'completed.json').read_text()));break
                    if 'out of memory' not in log.read_text(errors='replace').lower():
                        p.write_json(out/'runner-state.json',{'status':'failed_requires_diagnosis','task':task,'log':str(log),'exit_code':code,'at':p.now()});return code
                    minimum=min(gpu['total_mib']-1024,max(minimum,gpu['free_mib']+2048))
                    p.write_json(out/'runner-state.json',{'status':'oom_cooldown','task':task,'retry_not_before_unix':time.time()+1200,'at':p.now()});time.sleep(1200)
        p.write_json(out/'runner-state.json',{'status':'all_200_step_models_complete_evaluation_pending','at':p.now(),'next':'Evaluate each checkpoint against same-condition same-step retrain, then sequential requests.'})
    return 0

def main():
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    c=yaml.safe_load(Path(args.config).read_text());p=module(c['preparation_script'])
    if args.prepare:prepare(c,p);return 0
    if args.worker:worker(c,p);return 0
    return supervise(c,p)

if __name__=='__main__':sys.exit(main())
