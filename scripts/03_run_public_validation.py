"""Recompute current manuscript public tests and their Figure 3 sensitivity tables."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stablejoint import analyze_multiview,generate_signs
from stablejoint.benchmark import evaluate_published_baselines,evaluate_methods
from stablejoint.public_datasets import iter_public_subsets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensitivity',action='store_true',help='Also reproduce individual deletion, feature sampling and subgroup sensitivity')
    a=p.parse_args();config=json.loads((ROOT/'config/experiment.json').read_text());master=int(config['seed']);grid=tuple(config['k_grid'])
    out=ROOT/'results/public_validation';out.mkdir(parents=True,exist_ok=True)
    rows=[];methods=[];legacy=[];loo=[];sampled=[];seeds=[]
    rng=np.random.default_rng(master+7000)
    for ds,subset,label,x,y in iter_public_subsets(ROOT/'data/public_prepared'):
        n=len(x);blocks={'Microbiome':x,'Metabolome':y};common=dict(dataset_id=ds,subset=subset,display_label=label)
        signs,exact=generate_signs(n,max_random=255 if n>8 else 4095,seed=master+n)
        r=analyze_multiview(blocks,signs=signs,k_grid=grid)
        rows.append(dict(**common,paired_subjects=n,time_points=2,microbiome_features=x.shape[1],metabolome_features=y.shape[1],
                         sign_patterns=len(signs),exact_enumeration=exact,microbiome_p=r.block_results['Microbiome'].block_pvalue,
                         metabolome_p=r.block_results['Metabolome'].block_pvalue,joint_p=r.joint_pvalue))
        for method,value in {'AS-MVSPD':r.joint_pvalue,**evaluate_published_baselines(blocks,signs)}.items():
            methods.append(dict(data_scope='public',dataset=label,n_subjects=n,method=method,joint_p=value,sign_patterns=len(signs),full_group=exact))
        base,_=evaluate_methods(blocks,signs,k_grid=grid)
        for method,value in base.items():legacy.append(dict(**common,method=method,joint_p=value))
        for name,b in r.block_results.items():b.feature_table.to_csv(out/(ds+'_'+subset+'_'+name+'_features.csv'),index=False)
        if not a.sensitivity:continue
        sg,full=generate_signs(n-1,max_random=255 if n-1>8 else 4095,seed=master+7000+n-1)
        for omit in x.index:
            rr=analyze_multiview({k:b.drop(index=omit) for k,b in blocks.items()},signs=sg,k_grid=grid)
            loo.append(dict(**common,omitted_subject=omit,joint_p=rr.joint_pvalue,microbiome_p=rr.block_results['Microbiome'].block_pvalue,metabolome_p=rr.block_results['Metabolome'].block_pvalue,sign_patterns=len(sg),exact_enumeration=full))
        if ds.startswith('HE_'):rng_he=np.random.default_rng(master+40000)
        for rep in range(30):
            if ds.startswith('HE_'):
                bb={k:b.iloc[:,np.sort(rng_he.choice(b.shape[1],size=int(np.ceil(.8*b.shape[1])),replace=False))] for k,b in blocks.items()}
            else:
                bb={k:b.loc[:,rng.choice(b.columns.to_numpy(),size=max(1,int(np.ceil(.8*b.shape[1]))),replace=False)] for k,b in blocks.items()}
            rr=analyze_multiview(bb,signs=signs,k_grid=grid)
            for k,b in rr.block_results.items():sampled.append(dict(**common,replicate=rep,result_level=k,pvalue=b.block_pvalue,selected_k=b.selected_k))
            sampled.append(dict(**common,replicate=rep,result_level='Joint IUT',pvalue=rr.joint_pvalue,selected_k=np.nan))
        for i in range(1 if exact else 20):
            seed=master+7100+i
            sg,full=generate_signs(n,max_random=255 if n>8 else 4095,seed=seed)
            rr=analyze_multiview(blocks,signs=sg,k_grid=grid)
            seeds.append(dict(**common,seed_index=i,seed=seed,joint_p=rr.joint_pvalue,exact_enumeration=full))
    for name,data in [('public_dataset_overview.csv',rows),('Table_1_public_validation.csv',rows),('real_data_published_baselines.csv',methods),('public_method_comparison.csv',legacy)]:pd.DataFrame(data).to_csv(out/name,index=False)
    if a.sensitivity:
        for name,data in [('public_leave_one_subject.csv',loo),('public_feature_resampling.csv',sampled),('public_seed_sensitivity.csv',seeds)]:pd.DataFrame(data).to_csv(out/name,index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
