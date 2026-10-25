"""Sequential label-free requests and cumulative instruction retraining."""
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

SUBJECTS=['S01','S02','S03']
METHODS=['laf_importance_mask','ssd_only']

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def remaining(torch,overall,targets,total=167091,count=48):
    result={};clipped=0
    for name,mean in overall.items():
        numerator=mean.double()*total
        sub=sum((target[name].double()*count for target in targets),torch.zeros_like(numerator))
        tolerance=1e-6*(numerator.abs()+sub.abs())+1e-30
        raw=numerator-sub
        if (raw < -tolerance).any():raise ValueError('Material negative cumulative importance: '+name)
        clipped+=int((raw<0).sum());result[name]=(raw.clamp_min(0)/(total-count*len(targets))).float()
    return result,clipped

def prepare(c,p):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    assert p.digest(c['importance_path'])==c['importance_sha256']
    stats=torch.load(c['importance_path'],map_location='cpu',weights_only=True)
    with gzip.open(c['unlabeled_manifest'],'rt') as f:rows=json.load(f)
    assert len(rows)==167091 and all(set(row)=={'dataset','x_path','trial','subject','cache_row'} for row in rows)
    seq=Path(c['artifact_dir'])/'sequential';seq.mkdir(exist_ok=True)
    evidence={};tasks=[s['dataset'] for s in c['datasets']]
    for number,subject in enumerate(SUBJECTS[1:],1):
        old=SUBJECTS[:number];excluded=set(old+[subject])
        means,clipped=remaining(torch,stats['means']['overall'],[stats['means'][s] for s in old])
        mask={name:stats['means'][subject][name]>value for name,value in means.items()}
        folder=seq/'preparation'/subject;folder.mkdir(parents=True,exist_ok=True)
        torch.save({'means':{'overall':means,subject:stats['means'][subject]},'identity':{'original_importance_sha256':c['importance_sha256'],'previously_excluded':old,'remaining_count':167091-48*number,'target':subject,'target_count':48}},folder/'importance.pt')
        torch.save({'importance':mask},folder/'masks.pt')
        target=[i for i,row in enumerate(rows) if row['dataset']=='MI_Shin2017A' and row['subject']==subject];assert len(target)==48
        pools={ds:[i for i,row in enumerate(rows) if row['dataset']==ds and not(ds=='MI_Shin2017A' and row['subject'] in excluded)] for ds in tasks}
        assert sum(map(len,pools.values()))==167091-48*(number+1)
        rng=np.random.default_rng(42);pairs=[]
        for epoch in range(40):
            ff=rng.permutation(target).tolist();rr=[int(rng.choice(pools[tasks[(epoch*48+j)%17]])) for j in range(48)]
            for start in [0,32]:pairs.append({'epoch':epoch+1,'target':ff[start:start+32],'retained':rr[start:start+32]})
        assert len(pairs)==80
        for epoch in range(40):assert sorted(pairs[2*epoch]['target']+pairs[2*epoch+1]['target'])==target
        assert all(not(rows[i]['dataset']=='MI_Shin2017A' and rows[i]['subject'] in excluded) for pair in pairs for i in pair['retained'])
        pp=Path(c['run_dir'])/'splits'/('sequential-'+subject+'-pairs.json');p.write_json(pp,pairs)
        evidence[subject]={'importance_path':str(folder/'importance.pt'),'importance_sha256':p.digest(folder/'importance.pt'),'masks_path':str(folder/'masks.pt'),'masks_sha256':p.digest(folder/'masks.pt'),'paired_plan':str(pp),'paired_plan_sha256':p.digest(pp),'target_subject':subject,'previously_excluded':old,'remaining_count':167091-48*number,'retained_count':sum(map(len,pools.values())),'clipped_roundoff_elements':clipped,'selected_elements':sum(int(v.sum()) for v in mask.values()),'active_elements':sum(v.numel() for v in mask.values())}
    toy={'x':torch.tensor([1.,2.])};target={'x':torch.tensor([2.,1.])};got,_=remaining(torch,toy,[target],total=10,count=2)
    assert torch.allclose(got['x'],torch.tensor([.75,2.25]))
    p.write_json(Path(c['run_dir'])/'checks/sequential-preparation.json',{'at':p.now(),'requests':evidence,'synthetic_weighted_subtraction_passed':True,'label_free_manifest_only':True,'target_and_retainer_isolation_passed':True})

def worker(c,p):
    assert p.digest(c['laf_source'])==c['laf_source_sha256']
    laf=load(c['laf_source'],'sequential_laf')
    if c['task']=='cumulative_retrain':
        assert p.digest(c['initial_state'])==c['initial_state_sha256']
        p.retrain(c);return
    for field in ['importance','masks']:
        assert p.digest(c[field+'_path'])==c[field+'_sha256']
    write=p.write_json
    def redirected(path,value):
        if Path(path)==Path(c['artifact_dir'])/'immediate/progress.json':
            path=Path(c['artifact_dir'])/'sequential/progress.json';value.update(target_subject=c['target_subject'],request_number=SUBJECTS.index(c['target_subject'])+1)
        return write(path,value)
    p.write_json=redirected
    laf.worker(c,c['method'],p)
    out=Path(os.environ['LEAF_LAF_ATTEMPT_DIR']);result=json.loads((out/'completed.json').read_text());result.update(target_subject=c['target_subject'],request_number=SUBJECTS.index(c['target_subject'])+1,previous_checkpoint=c['original_checkpoint'],previous_sha256=c['original_sha256'],previously_excluded=c['previously_excluded'],remaining_count=c['remaining_count'],retained_count=c['retained_count'],reference_status='no dedicated two-subject retrain' if c['target_subject']=='S02' else 'three-subject retrain pending')
    p.write_json(out/'completed.json',result)

def supervise(c,p):
    import yaml,fcntl
    seq=Path(c['artifact_dir'])/'sequential';run=Path(c['run_dir']);ready=json.loads((run/'checks/sequential-preparation.json').read_text())['requests']
    with (seq/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=[method+'--'+subject for subject in SUBJECTS[1:] for method in METHODS]+['cumulative_retrain']
        for task in tasks:
            marker=seq/'results'/(task+'.json')
            if marker.exists():continue
            count=len(list((seq/'attempts'/task).glob('attempt*')));minimum=28000 if task=='cumulative_retrain' else 9000
            while True:
                gpu=max(p.gpu_inventory(),key=lambda g:g['free_mib'])
                if gpu['free_mib']<minimum:
                    p.write_json(seq/'runner-state.json',{'status':'waiting_gpu','task':task,'minimum_mib':minimum,'gpu':gpu,'at':p.now()});time.sleep(60);continue
                count+=1;out=seq/'attempts'/task/f'attempt{count:03d}';out.mkdir(parents=True)
                cfg=copy.deepcopy(c);cfg.update(task=task,attempt=count)
                if task=='cumulative_retrain':
                    cfg.update(c['cumulative_retrain']);cfg['artifact_dir']=str(seq/'retrain_reference');Path(cfg['artifact_dir']).mkdir(exist_ok=True)
                else:
                    method,subject=task.split('--');cfg.update(ready[subject]);cfg.update(method=method,target_vae=c['target_vaes'][subject]);previous=SUBJECTS[SUBJECTS.index(subject)-1]
                    old=json.loads((seq/'results'/(method+'--'+previous+'.json')).read_text());cfg.update(original_checkpoint=old['checkpoint'],original_sha256=old['sha256'])
                path=run/'configs/sequential'/task/f'attempt{count:03d}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(path),'--worker']
                cfg['execution'].update(command=command,physical_gpu=gpu['index'],gpu_uuid=gpu['uuid'],CUDA_VISIBLE_DEVICES=gpu['uuid'])
                path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));mp=run/'manifest.json';m=json.loads(mp.read_text());m.setdefault('sequential_configs',[]).append({'path':str(path),'sha256':p.digest(path)});p.write_json(mp,m)
                log=run/'logs'/f'sequential-{task}-attempt{count:03d}.log'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],LEAF_LAF_ATTEMPT_DIR=str(out),LEAF_STAGE_ATTEMPT_DIR=str(out),OMP_NUM_THREADS='8')
                with log.open('w') as f:
                    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
                    p.write_json(seq/'runner-state.json',{'status':'running','task':task,'child_pid':child.pid,'supervisor_pid':os.getpid(),'gpu':gpu,'log':str(log),'config':str(path),'at':p.now()});code=child.wait()
                log.with_suffix('.exitcode').write_text(str(code)+'\n')
                if code==0:
                    result=json.loads((Path(cfg['artifact_dir'])/'retrain_instruction/completed.json' if task=='cumulative_retrain' else out/'completed.json').read_text());result.update(config=str(path),config_sha256=p.digest(path),exit_code=0);p.write_json(marker,result);break
                if 'out of memory' not in log.read_text(errors='replace').lower():
                    p.write_json(seq/'runner-state.json',{'status':'failed_requires_diagnosis','task':task,'exit_code':code,'log':str(log),'at':p.now()});return code
                minimum=min(gpu['total_mib']-1024,max(minimum,gpu['free_mib']+2048));p.write_json(seq/'runner-state.json',{'status':'oom_cooldown','task':task,'retry_not_before_unix':time.time()+1200,'at':p.now()});time.sleep(1200)
        p.write_json(seq/'runner-state.json',{'status':'sequential_weights_and_cumulative_retrain_complete_evaluation_pending','at':p.now(),'next':'evaluate each request on every forgotten subject and retained tasks; second request has no dedicated retrain reference'})
    return 0

def main():
    import yaml
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');args=ap.parse_args();c=yaml.safe_load(Path(args.config).read_text());p=load(c['preparation_script'],'sequential_preparation')
    assert p.digest(c['preparation_script'])==c['preparation_source_sha256']
    if args.prepare:prepare(c,p);return 0
    if args.worker:worker(c,p);return 0
    return supervise(c,p)

if __name__=='__main__':sys.exit(main())
