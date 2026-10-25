"""Sealed-candidate evaluation of instruction-level unlearning experiments."""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VIEWS = ["task_blended", "default", "task_unblended"]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def jsd(p, q):
    import numpy as np
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = (p+q)/2
    tiny = np.finfo(np.float64).tiny
    terms = .5*(p*np.log(np.clip(p, tiny, None)/np.clip(m, tiny, None)) +
                q*np.log(np.clip(q, tiny, None)/np.clip(m, tiny, None)))
    return np.maximum(terms.sum(-1), 0.)


def metrics(labels, probabilities):
    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    labels = np.asarray(labels, dtype=int)
    pred = probabilities.argmax(1)
    k = probabilities.shape[1]
    cm = confusion_matrix(labels, pred, labels=list(range(k)))
    supports = cm.sum(1)
    recalls = [float(cm[i,i]/supports[i]) if supports[i] else None for i in range(k)]
    kap = float(cohen_kappa_score(labels, pred))
    auc = None
    if len(np.unique(labels))==k:
        auc = float(roc_auc_score(labels, probabilities[:,1])) if k==2 else float(roc_auc_score(labels, probabilities.astype(float)/probabilities.sum(1,keepdims=True),multi_class="ovr"))
    return {"n":len(labels), "balanced_accuracy":float(balanced_accuracy_score(labels,pred)),
            "accuracy":float(accuracy_score(labels,pred)), "cohen_kappa":kap if np.isfinite(kap) else None,
            "weighted_f1":float(f1_score(labels,pred,average="weighted",zero_division=0)),
            "recall_by_class":recalls,"support_by_class":supports.tolist(),"confusion_matrix":cm.tolist(),
            "missing_true_classes":[i for i,n in enumerate(supports) if n==0],"auroc":auc,
            "cross_entropy":float(-np.log(np.clip(probabilities[np.arange(len(labels)),labels],1e-30,None)).mean())}


def prepare(config, prep):
    import h5py
    import numpy as np
    run, art = Path(config["run_dir"]), Path(config["artifact_dir"])
    bank = json.loads(Path(config["candidate_bank"]).read_text())
    assert len(bank["models"])==9
    assert all(prep.digest(v["checkpoint"])==v["sha256"] for v in bank["models"].values())
    with gzip.open(config["index_path"],"rt") as f:
        index=json.load(f)
    groups={}
    for spec in config["datasets"]:
        ds=spec["dataset"]
        expected=spec["current_file_identity"]
        st=Path(spec["path"]).stat()
        assert (st.st_size,st.st_mtime_ns)==(expected["bytes"],expected["mtime_ns"])
        valid=[row for row in index["valid"] if row[0]==ds and not(ds=="MI_Shin2017A" and row[4]=="S01")]
        groups["valid__"+ds]=valid
        test=[]
        with h5py.File(spec["path"],"r",locking=False) as f:
            if "testX" in f:
                y=np.asarray(f["testY"][:]).reshape(-1)
                test=[[ds,"testX",i,int(label),"predefined"] for i,label in enumerate(y)]
            elif ds=="CS_BCIC_Speech":
                for subject in f.keys():
                    y=np.asarray(f[subject]["Y"][300:]).reshape(-1)
                    test.extend([ds,subject+"/X",i+300,int(label),subject] for i,label in enumerate(y))
            else:
                for subject in spec["split"]["test_groups"]:
                    subject=str(subject)
                    labels=np.asarray(f[subject]["Y"][:]).reshape(-1)
                    test.extend([ds,subject+"/X",i,int(y),subject] for i,y in enumerate(labels) if not(ds=="MI_HighGamma" and y==2))
        assert len(test)==spec["split"]["test"]["n"],ds
        groups["test__"+ds]=test
    shin=[row for row in index["train"] if row[0]=="MI_Shin2017A"]
    groups["target_train"]=[row for row in shin if row[4]=="S01"]
    groups["retain_train__MI_Shin2017A"]=[row for row in shin if row[4]!="S01"]
    target_valid=[row.copy() for row in index["valid"] if row[0]=="MI_Shin2017A" and row[4]=="S01"]
    path=next(x["path"] for x in config["datasets"] if x["dataset"]=="MI_Shin2017A")
    with h5py.File(path,"r",locking=False) as f:
        for row in target_valid:row[3]=int(np.asarray(f[row[4]]["Y"][row[2]]).item())
    groups["target_valid"]=target_valid
    assert len(groups["target_train"])==48 and len(target_valid)==12
    assert len(groups["retain_train__MI_Shin2017A"])==1008 and len(groups["valid__MI_Shin2017A"])==252
    rng=np.random.default_rng(42)
    subset=[]
    for ds in [s["dataset"] for s in config["datasets"]]:
        choices=[i for i,row in enumerate(index["train"]) if row[0]==ds and not(ds=="MI_Shin2017A" and row[4]=="S01")]
        subset.extend(sorted(rng.choice(choices,size=min(64,len(choices)),replace=False).tolist()))
    groups["repr_retain"]=[index["train"][i] for i in subset]
    out=Path(config["evaluation_manifest"])
    with gzip.open(out,"wt") as f:json.dump(groups,f,separators=(",",":"))
    subjects=list(dict.fromkeys(row[4] for row in groups["retain_train__MI_Shin2017A"]))
    assert len(subjects)==21
    train,valid=groups["retain_train__MI_Shin2017A"],groups["valid__MI_Shin2017A"]
    attack={"fit_subjects":subjects[:14],"check_subjects":subjects[14:]}
    for split,chosen in (("fit",subjects[:14]),("check",subjects[14:])):
        attack[split+"_train"]=[i for i,row in enumerate(train) if row[4] in chosen]
        attack[split+"_valid"]=[i for i,row in enumerate(valid) if row[4] in chosen]
    assert tuple(len(attack[k]) for k in ("fit_train","fit_valid","check_train","check_valid"))==(672,168,336,84)
    for name,members,nonmembers,mi,ni in (("fit",train,valid,attack["fit_train"],attack["fit_valid"]),
                                      ("check",train,valid,attack["check_train"],attack["check_valid"]),
                                      ("target",groups["target_train"],target_valid,list(range(48)),list(range(12)))):
        mm,nn=[],[]
        for label in (0,1):
            a=[i for i in mi if members[i][3]==label];b=[i for i in ni if nonmembers[i][3]==label];n=min(len(a),len(b))
            mm.extend(rng.choice(a,size=n,replace=False).tolist());nn.extend(rng.choice(b,size=n,replace=False).tolist())
        attack[name+"_matched_train"],attack[name+"_matched_valid"]=mm,nn
    prep.write_json(run/"splits/attack-plan.json",attack)
    prep.write_json(run/"splits/representation-subset.json",{"original_train_indices":subset,"per_task":64,"seed":42,
                    "selected_before_any_candidate_predictions":True,"exclude_target_subject":"S01"})
    prep.write_json(run/"checks/evaluation-data-ready.json",{"at":prep.now(),"candidate_seal":config["candidate_bank"],
                    "candidate_seal_sha256":prep.digest(config["candidate_bank"]),"counts":{k:len(v) for k,v in groups.items()},
                    "evaluation_manifest_sha256":prep.digest(out),"attack_plan_sha256":prep.digest(run/"splits/attack-plan.json"),
                    "representation_subset_sha256":prep.digest(run/"splits/representation-subset.json")})


class Inputs:
    def __init__(self,config,text,hist,prep):
        self.paths={s["dataset"]:s["path"] for s in config["datasets"]}
        self.vectors={};self.prototypes={};self.files={};self.arrays={}
        import numpy as np
        for ds,classes in hist.Targets.items():
            q=text[hist.Instructions[prep.task_family(ds)][0]]
            candidate=np.mean([text[t] for t in classes],axis=0);candidate/=np.linalg.norm(candidate)
            blend=(q+candidate)/2;blend/=np.linalg.norm(blend)
            self.vectors[ds]={"task_blended":blend,"default":text["None"],"task_unblended":q}
            self.prototypes[ds]=np.stack([text[t] for t in classes]).astype("float32")

    def batch(self,records,view,np,torch):
        import h5py
        xs,qs=[],[]
        for ds,path,i,_,_ in records:
            if ds not in self.files:self.files[ds]=h5py.File(self.paths[ds],"r",locking=False)
            key=(ds,path)
            if key not in self.arrays:self.arrays[key]=self.files[ds][path]
            x=np.asarray(self.arrays[key][i],dtype=np.float32)
            assert np.isfinite(x).all()
            x=np.pad(x,((0,0),(0,2000-x.shape[1]))) if x.shape[1]<2000 else x[:,:2000]
            xs.append(x);qs.append(self.vectors[ds][view])
        return torch.from_numpy(np.stack(xs)).cuda(),torch.from_numpy(np.stack(qs)).cuda()


def context(config,prep,identity="original"):
    np,torch,_,hist=prep.runtime(config)
    assert prep.digest(config["candidate_bank"])==config["candidate_bank_sha256"]
    assert prep.digest(config["evaluation_manifest"])==config["evaluation_manifest_sha256"]
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    spec=config["models"][identity]
    assert prep.digest(spec["checkpoint"])==spec["sha256"]
    model=hist.LEAF(hist.build_model_config(hist.load_yaml()))
    model.load_state_dict(torch.load(spec["checkpoint"],map_location="cpu",weights_only=True),strict=True)
    model.cuda().eval().requires_grad_(False)
    reader=Inputs(config,hist.load_embeddings("mpnet-base"),hist,prep)
    with gzip.open(config["evaluation_manifest"],"rt") as f:groups=json.load(f)
    return np,torch,model,reader,groups


def embeddings(records,view,model,reader,np,torch):
    values=[]
    with torch.inference_mode():
        for start in range(0,len(records),32):
            x,q=reader.batch(records[start:start+32],view,np,torch)
            e,_=model(x,q)
            assert torch.isfinite(e).all()
            values.append(e.cpu().numpy())
    return np.concatenate(values)


def preflight_cost(config,prep):
    start=time.monotonic()
    np,torch,model,reader,groups=context(config,prep)
    laf=module(config["laf_source"],"laf_cost_replay")
    masks=torch.load(config["masks_path"],map_location="cpu",weights_only=True)["importance"]
    masks={n:(masks[n].cuda() if n in masks else torch.zeros_like(p,dtype=torch.bool)) for n,p in model.named_parameters()}
    stats=torch.load(config["importance_path"],map_location="cpu",weights_only=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();operation=time.monotonic()
    laf.dampen(torch,model,masks,stats["means"]["overall"],stats["means"]["S01"])
    torch.cuda.synchronize();scale_seconds=time.monotonic()-operation;request_seconds=time.monotonic()-start
    peak_allocated=torch.cuda.max_memory_allocated()/2**20;peak_reserved=torch.cuda.max_memory_reserved()/2**20
    damped=torch.load(config["models"]["ssd_only"]["checkpoint"],map_location="cpu",weights_only=True)
    assert all(torch.equal(value.cpu(),damped[k]) for k,value in model.state_dict().items())
    prep.write_json(Path(config["artifact_dir"])/"evaluation/ssd-cost-correction.json",{
        "at":prep.now(),"one_shot_scaling_gpu_synchronized_seconds":scale_seconds,
        "replay_harness_setup_through_scaling_seconds":request_seconds,
        "peak_allocated_mib":peak_allocated,"peak_reserved_mib":peak_reserved,
        "all_replayed_tensors_match_sealed_SSD":True,"sealed_SSD_sha256":config["models"]["ssd_only"]["sha256"],
        "applies_to":["ssd_only","ssd_then_laf_mask pre-damp component"],
        "raw_immediate_cost_replaced_for_reporting_only":True})
    model.load_state_dict(torch.load(config["models"]["original"]["checkpoint"],map_location="cpu",weights_only=True),strict=True)
    repeats={}
    for view in VIEWS:
        first=embeddings(groups["target_train"],view,model,reader,np,torch)
        second=embeddings(groups["target_train"],view,model,reader,np,torch)
        proto=reader.prototypes["MI_Shin2017A"]
        p=torch.softmax(torch.from_numpy(first@proto.T),dim=-1).numpy()
        q=torch.softmax(torch.from_numpy(second@proto.T),dim=-1).numpy()
        diff=float(np.max(np.abs(p-q)));assert diff<=1e-6
        repeats[view]={"mean_repeated_prediction_JSD":float(jsd(p,q).mean()),"max_probability_difference":diff}
    epsilon=max(1e-8,5*max(v["mean_repeated_prediction_JSD"] for v in repeats.values()))
    prep.write_json(Path(config["run_dir"])/"checks/numerical-tolerance.json",{
        "at":prep.now(),"epsilon":epsilon,"repeated_original":repeats,"trials":48,
        "defined_before_formal_candidate_evaluation":True,"heldout_or_test_used_for_tolerance":False})


def predict(config,prep,identity):
    np,torch,model,reader,groups=context(config,prep,identity)
    out=Path(os.environ["LEAF_EVAL_ATTEMPT_DIR"]);out.mkdir(parents=True,exist_ok=True)
    assert Path(config["run_dir"],"checks/numerical-tolerance.json").exists()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    outputs={}
    for view in VIEWS:
        folder=out/view;folder.mkdir(exist_ok=True)
        summary={};representations={}
        for name,records in groups.items():
            e=embeddings(records,view,model,reader,np,torch)
            if name=="repr_retain":
                representations[name]=e
                continue
            ds=records[0][0]
            logits=e@reader.prototypes[ds].T
            probability=torch.softmax(torch.from_numpy(logits),dim=-1).numpy()
            y=np.array([row[3] for row in records],dtype=np.int64)
            assert np.all((y>=0)&(y<probability.shape[1]))
            subjects=np.array([row[4] for row in records])
            ids=np.array([f"{row[0]}:{row[1]}:{row[2]}" for row in records])
            entropy=-(probability*np.log(np.clip(probability,1e-30,None))).sum(1)
            path=folder/(name+".npz")
            np.savez_compressed(path,y=y,pred=probability.argmax(1),logits=logits,probabilities=probability,
                                entropy=entropy,subjects=subjects,trial_ids=ids)
            score=metrics(y,probability)
            norms=np.linalg.norm(e,axis=1)
            score['embedding_norm_min']=float(norms.min());score['embedding_norm_max']=float(norms.max())
            if name.startswith("test__"):
                score["per_subject"]={subject:metrics(y[subjects==subject],probability[subjects==subject]) for subject in dict.fromkeys(subjects)}
            summary[name]=score
            if name in ("target_train","target_valid"):representations[name]=e
            prep.write_json(Path(config["artifact_dir"])/"evaluation/progress.json",{
                "at":prep.now(),"model":identity,"view":view,"group":name,"n":len(records)})
            print(identity,view,name,len(records),flush=True)
        np.savez_compressed(folder/"representations.npz",**representations)
        prep.write_json(folder/"metrics.json",summary)
        outputs[view]={"directory":str(folder),"metrics_sha256":prep.digest(folder/"metrics.json"),
                       "representations_sha256":prep.digest(folder/"representations.npz")}
    torch.cuda.synchronize()
    prep.write_json(out/"completed.json",{"model":identity,"at":prep.now(),"input":config["models"][identity],"views":outputs,
        "elapsed_gpu_synchronized_seconds":time.monotonic()-started,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20})


def attack_score(truth,score):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    if len(np.unique(truth))<2:return {"n":len(truth),"balanced_accuracy":None,"roc_auc":None}
    return {"n":len(truth),"balanced_accuracy":float(balanced_accuracy_score(truth,score>=.5)),"roc_auc":float(roc_auc_score(truth,score))}


def member_attack(folder,plan):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    train=np.load(folder/"retain_train__MI_Shin2017A.npz",allow_pickle=False)
    valid=np.load(folder/"valid__MI_Shin2017A.npz",allow_pickle=False)
    tt=np.load(folder/"target_train.npz",allow_pickle=False);tv=np.load(folder/"target_valid.npz",allow_pickle=False)
    fit=np.r_[train['entropy'][plan['fit_train']],valid['entropy'][plan['fit_valid']]][:,None]
    y=np.r_[np.ones(len(plan['fit_train'])),np.zeros(len(plan['fit_valid']))]
    model=LogisticRegression(C=1,class_weight="balanced",solver="lbfgs",max_iter=1000,random_state=42).fit(fit,y)
    result={"feature":"predictive entropy", "coef":model.coef_.tolist(),"intercept":model.intercept_.tolist(),"threshold":.5}
    for name,members,nonmembers,mi,ni in (("check",train,valid,plan['check_train'],plan['check_valid']),
                                      ("target",tt,tv,list(range(48)),list(range(12))),
                                      ("check_matched",train,valid,plan['check_matched_train'],plan['check_matched_valid']),
                                      ("target_matched",tt,tv,plan['target_matched_train'],plan['target_matched_valid'])):
        truth=np.r_[np.ones(len(mi)),np.zeros(len(ni))]
        scores=model.predict_proba(np.r_[members['entropy'][mi],nonmembers['entropy'][ni]][:,None])[:,1]
        classes=np.r_[members['y'][mi],nonmembers['y'][ni]]
        result[name]=attack_score(truth,scores)
        result[name]['by_true_class']={str(k):attack_score(truth[classes==k],scores[classes==k]) for k in (0,1)}
    return result


def cka(x,y):
    import numpy as np
    x,y=x.astype(float)-x.mean(0),y.astype(float)-y.mean(0)
    numerator=np.linalg.norm(x.T@y,'fro')**2
    denominator=np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')
    return float(numerator/denominator) if denominator>0 else None


def summarize(config,prep):
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run,art=Path(config['run_dir']),Path(config['artifact_dir'])
    results={k:json.loads((art/'evaluation/results'/(k+'.json')).read_text()) for k in config['models']}
    attack=json.loads((run/'splits/attack-plan.json').read_text())
    epsilon=json.loads((run/'checks/numerical-tolerance.json').read_text())['epsilon']
    folders={k:{v:Path(result['views'][v]['directory']) for v in VIEWS} for k,result in results.items()}
    original_repr=np.load(folders['original']['task_blended']/'representations.npz',allow_pickle=False)
    pca=PCA(n_components=2,svd_solver='full').fit(original_repr['repr_retain'])
    output=art/'evaluation/summary';output.mkdir(exist_ok=True)
    np.savez_compressed(output/'fixed-pca.npz',components=pca.components_,mean=pca.mean_,explained_variance_ratio=pca.explained_variance_ratio_)
    projected=[]
    for model_folders in folders.values():
        for folder in model_folders.values():
            with np.load(folder/'representations.npz',allow_pickle=False) as z:
                projected.extend(pca.transform(z[g]) for g in ('repr_retain','target_train','target_valid'))
    bounds=np.concatenate(projected);low=bounds.min(0);high=bounds.max(0);padding=np.maximum((high-low)*.05,.001)
    all_results={};rows=[]
    for view in VIEWS:
        by_model={}
        originals=json.loads((folders['original'][view]/'metrics.json').read_text())
        base_test={k:v['balanced_accuracy'] for k,v in originals.items() if k.startswith('test__')}
        common=[k for k in base_test if k!='test__ADHD_AliMotie']
        assert len(common)==16
        original_overall=float(np.mean([base_test[k] for k in common]))
        reference_probs={g:np.load(folders['retrain_instruction'][view]/(g+'.npz'),allow_pickle=False) for g in ('target_train','target_valid')}
        original_dist={g:float(jsd(np.load(folders['original'][view]/(g+'.npz'),allow_pickle=False)['probabilities'],reference_probs[g]['probabilities']).mean()) for g in reference_probs}
        fig,axes=plt.subplots(3,3,figsize=(13,11),sharex=True,sharey=True)
        for (identity,folder),ax in zip(((k,folders[k][view]) for k in results),axes.flat):
            score=json.loads((folder/'metrics.json').read_text())
            overall=float(np.mean([score[k]['balanced_accuracy'] for k in common]))
            aggregate={'Overall16_test_bacc':overall,'ADHD_test_bacc':score['test__ADHD_AliMotie']['balanced_accuracy'],
                       'Overall16_delta_vs_original_pp':100*(overall-original_overall)}
            target={}
            for group,ref in reference_probs.items():
                current=np.load(folder/(group+'.npz'),allow_pickle=False)
                distances=jsd(current['probabilities'],ref['probabilities'])
                target[group]={'mean_JSD_vs_retrain':float(distances.mean()),'prediction_disagreement_vs_retrain':float(np.mean(current['pred']!=ref['pred'])),
                               'JSD_reduction_vs_original':original_dist[group]-float(distances.mean()),
                               'relative_JSD_reduction':(original_dist[group]-float(distances.mean()))/original_dist[group] if original_dist[group]>epsilon else None}
                np.savez_compressed(folder/(group+'-vs-retrain.npz'),trial_ids=current['trial_ids'],JSD=distances,
                                    disagreements=current['pred']!=ref['pred'])
            representations=np.load(folder/'representations.npz',allow_pickle=False)
            old=np.load(folders['original'][view]/'representations.npz',allow_pickle=False)
            ref=np.load(folders['retrain_instruction'][view]/'representations.npz',allow_pickle=False)
            repr_metrics={}
            projection={}
            for group in ('repr_retain','target_train','target_valid'):
                x=representations[group]
                repr_metrics[group]={'linear_CKA_vs_original':cka(x,old[group]),'linear_CKA_vs_retrain':cka(x,ref[group]),
                    'mean_cosine_distance_vs_original':float(np.mean(1-np.clip(np.sum(x*old[group],1),-1,1)))}
                projection[group]=pca.transform(x)
                color={'repr_retain':'#aaaaaa','target_train':'#d62728','target_valid':'#1f77b4'}[group]
                ax.scatter(projection[group][:,0],projection[group][:,1],s=3 if group=='repr_retain' else 12,alpha=.35 if group=='repr_retain' else .8,c=color,label=group)
            ax.set_title(identity,fontsize=9)
            ax.set_xlim(low[0]-padding[0],high[0]+padding[0]);ax.set_ylim(low[1]-padding[1],high[1]+padding[1])
            np.savez_compressed(folder/'fixed-pca-projections.npz',**projection)
            per_task_drop={k:score[k]['balanced_accuracy']-base_test[k] for k in base_test}
            criteria={'original_train_gap_measurable':original_dist['target_train']>epsilon,
                      'train_JSD_at_most_90pct':target['target_train']['mean_JSD_vs_retrain']<=.9*original_dist['target_train'] if original_dist['target_train']>epsilon else None,
                      'heldout_JSD_within_tolerance':target['target_valid']['mean_JSD_vs_retrain']<=original_dist['target_valid']+max(epsilon,.05*original_dist['target_valid']),
                      'Shin_test_drop_at_most_2pp':per_task_drop['test__MI_Shin2017A']>=-.02,
                      'Overall16_test_drop_at_most_2pp':overall-original_overall>=-.02,
                      'task_drops_over_5pp':[k for k,v in per_task_drop.items() if v<-.05]}
            criteria['overall_engineering_pass']=all(criteria[k] is True for k in ('original_train_gap_measurable','train_JSD_at_most_90pct','heldout_JSD_within_tolerance','Shin_test_drop_at_most_2pp','Overall16_test_drop_at_most_2pp'))
            by_model[identity]={'metrics':score,'aggregate':aggregate,'target_closeness':target,'representations':repr_metrics,
                                'membership_inference':member_attack(folder,attack),'criteria':criteria}
            rows.append({'view':view,'model':identity,'target_train_bacc':score['target_train']['balanced_accuracy'],
                         'target_valid_bacc':score['target_valid']['balanced_accuracy'],'target_train_JSD':target['target_train']['mean_JSD_vs_retrain'],
                         'target_valid_JSD':target['target_valid']['mean_JSD_vs_retrain'],'Shin_test_bacc':score['test__MI_Shin2017A']['balanced_accuracy'],
                         'Overall16_test_bacc':overall,'engineering_pass':criteria['overall_engineering_pass']})
        fig.suptitle(view+' / one fixed PCA fitted on original retained training subset')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
        fig.tight_layout(rect=(0,.025,1,.97));fig.savefig(output/(view+'-pca.png'),dpi=180);fig.savefig(output/(view+'-pca.pdf'));plt.close(fig)
        all_results[view]={'epsilon':epsilon,'original_JSD_vs_retrain':original_dist,'models':by_model}
    prep.write_json(output/'comparison.json',all_results)
    import csv
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    prep.write_json(output/'completed.json',{'at':prep.now(),'models':9,'views':VIEWS,'comparison_sha256':prep.digest(output/'comparison.json'),
                    'candidate_seal_sha256':prep.digest(config['candidate_bank']),'all_candidate_weights_fixed_before_evaluation':True})


def synthetic(config,prep):
    import numpy as np
    p=np.array([[.2,.8],[.7,.3]])
    assert np.array_equal(jsd(p,p),np.zeros(2))
    assert np.isclose(jsd(np.array([[1.,0.]]),np.array([[0.,1.]]))[0],np.log(2))
    y=np.array([0]*90+[1]*10);pred=np.array([0]*80+[1]*10+[0]*5+[1]*5)
    probs=np.eye(2)[pred]
    m=metrics(y,probs)
    assert np.isclose(m['balanced_accuracy'],(.8888888888888888+.5)/2)
    assert np.isclose(m['cohen_kappa'],.3181818181818182)
    x=np.random.default_rng(42).normal(size=(12,6));assert np.isclose(cka(x,x),1.)
    prep.write_json(Path(config['run_dir'])/'checks/evaluation-synthetic.json',{'at':prep.now(),'JSD_self_and_disjoint':'passed','project_confusion_matrix_example':'passed','CKA_identity':'passed','real_data_used':False})


def supervise(config,prep):
    import fcntl,yaml
    run,art=Path(config['run_dir']),Path(config['artifact_dir']);out=art/'evaluation';out.mkdir(exist_ok=True)
    with (out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        for task in ['cost_and_tolerance']+list(config['models'])+['summarize']:
            marker=out/'results'/(task+'.json')
            if marker.exists():continue
            attempt=len(list((out/'attempts'/task).glob('attempt*')));minimum=5000
            while True:
                gpu=max(prep.gpu_inventory(),key=lambda x:x['free_mib'])
                if task!='summarize' and gpu['free_mib']<minimum:
                    prep.write_json(out/'runner-state.json',{'status':'waiting_gpu','task':task,'gpu':gpu,'at':prep.now()});time.sleep(60);continue
                attempt+=1;folder=out/'attempts'/task/f'attempt{attempt:03d}';folder.mkdir(parents=True)
                effective=copy.deepcopy(config);effective.update(task=task,attempt=attempt)
                conf=run/'configs/evaluation'/task/f'attempt{attempt:03d}.yaml';conf.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(conf),'--task',task]
                effective['execution'].update(command=command,physical_gpu=None if task=='summarize' else gpu['index'],gpu_uuid=None if task=='summarize' else gpu['uuid'],CUDA_VISIBLE_DEVICES='' if task=='summarize' else gpu['uuid'])
                conf.write_text(yaml.safe_dump(effective,allow_unicode=True,sort_keys=False))
                mp=run/'manifest.json';m=json.loads(mp.read_text());m.setdefault('evaluation_configs',[]).append({'path':str(conf),'sha256':prep.digest(conf)});prep.write_json(mp,m)
                log=run/'logs'/f'evaluation-{task}-attempt{attempt:03d}.log'
                with log.open('w') as f:
                    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if task=='summarize' else gpu['uuid'],OMP_NUM_THREADS='8',LEAF_EVAL_ATTEMPT_DIR=str(folder)))
                    prep.write_json(out/'runner-state.json',{'status':'running','task':task,'child_pid':child.pid,'supervisor_pid':os.getpid(),'gpu':gpu,'at':prep.now(),'log':str(log),'config':str(conf)})
                    code=child.wait()
                (run/'logs'/f'evaluation-{task}-attempt{attempt:03d}.exitcode').write_text(str(code)+'\n')
                if code==0:
                    result=json.loads((folder/'completed.json').read_text()) if (folder/'completed.json').exists() else {'task':task,'at':prep.now()}
                    result.update(exit_code=0,config=str(conf),config_sha256=prep.digest(conf));prep.write_json(marker,result);break
                if 'out of memory' not in log.read_text(errors='replace').lower():
                    prep.write_json(out/'runner-state.json',{'status':'failed_requires_diagnosis','task':task,'exit_code':code,'log':str(log),'at':prep.now()});return code
                minimum=min(gpu['total_mib']-1024,max(minimum,gpu['free_mib']+2048))
                prep.write_json(out/'runner-state.json',{'status':'oom_cooldown','task':task,'retry_not_before_unix':time.time()+1200,'at':prep.now()});time.sleep(1200)
        prep.write_json(out/'runner-state.json',{'status':'unified_evaluation_complete','at':prep.now(),'next':'Read comparison, report limitations, then execute all precommitted continued-learning and sequential-request stages.'})
        return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--synthetic',action='store_true');p.add_argument('--task');args=p.parse_args()
    import yaml
    config=yaml.safe_load(args.config.read_text());prep=module(config['preparation_script'],'preparation_eval')
    if args.prepare:prepare(config,prep);return 0
    if args.synthetic:synthetic(config,prep);return 0
    if args.task=='cost_and_tolerance':preflight_cost(config,prep);return 0
    if args.task=='summarize':summarize(config,prep);return 0
    if args.task:predict(config,prep,args.task);return 0
    return supervise(config,prep)


if __name__=='__main__':sys.exit(main())
