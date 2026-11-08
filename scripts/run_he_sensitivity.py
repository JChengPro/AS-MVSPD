import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys,json,hashlib,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/he_validation'
OUT=BASE/'further_audit'; OUT.mkdir(parents=True,exist_ok=True)

sys.path.insert(0,str(ROOT))
from stablejoint import analyze_multiview,generate_signs
from stablejoint.preprocessing import _paired_changes,clr_transform
config=json.loads((ROOT/'config/experiment.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csv(n,v): pd.DataFrame(v).to_csv(OUT/n,index=False,encoding='utf-8-sig')

save('audit_plan.json',{'fixed_comparison':'2 to 12 months','analyses':['pair and numeric QC','all three feeding strata with within-stratum preprocessing','57 individual deletions, locked features','30 independent 80-percent feature subsets','metabolite raw, asinh, per-feature half-minimum-positive log, total-relative log, high confidence, prevalence >=25%, no urea','microbiome prevalence 10%,50%; remove Unclassified; pseudocount sensitivity','exclude pairs with any positive Urea','4096-element closed subgroup','descriptive skewness; cannot prove sign invariance'],'seed':int(config['seed']),'no_selection':'Report all specified sensitivity results; primary result remains unchanged'})
raw=ROOT/'data/raw/borenstein_curated/HE_INFANTS_MFGM_2019'
meta=pd.read_csv(raw/'metadata.tsv',sep='\t'); micro=pd.read_csv(raw/'genera.tsv',sep='\t',index_col=0); metab=pd.read_csv(raw/'mtb.tsv',sep='\t',index_col=0)
pairs=pd.read_csv(ROOT/'data/public_prepared/HE_INFANTS_MFGM_2019/pair_mapping.csv',dtype={'subject_id':str})
x=pd.read_csv(ROOT/'data/public_prepared/HE_INFANTS_MFGM_2019/microbiome_paired_changes.csv',index_col=0); y=pd.read_csv(ROOT/'data/public_prepared/HE_INFANTS_MFGM_2019/metabolome_paired_changes.csv',index_col=0)
x.index=x.index.astype(str); y.index=y.index.astype(str)
assert x.index.equals(y.index) and pairs.subject_id.tolist()==x.index.tolist()
assert meta.Sample.is_unique and micro.index.is_unique and metab.index.is_unique
lookup=meta.set_index('Sample')
for r in pairs.itertuples():
    a=lookup.loc[r.pre_sample]; b=lookup.loc[r.post_sample]
    assert str(a.Subject)==str(b.Subject)==r.subject_id and a.Age==2 and b.Age==12
samples=pairs.pre_sample.tolist()+pairs.post_sample.tolist()
M=metab.loc[samples]; G=micro.loc[samples]
quality={'paired_n':len(pairs),'unique_samples':len(set(samples)),'same_gender_within_pair':all(lookup.loc[r.pre_sample,'Gender']==lookup.loc[r.post_sample,'Gender'] for r in pairs.itertuples()),'metabolite_missing_cells':int(M.isna().sum().sum()),'metabolite_zero_fraction':float((M==0).to_numpy().mean()),'negative_microbe_cells':int((G<0).sum().sum()),'negative_metabolite_cells':int((M<0).sum().sum()),'microbiome_row_sum_range':[float(G.sum(axis=1).min()),float(G.sum(axis=1).max())],'urea_positive_samples':int((M['Urea']>0).sum()),'urea_max':float(M['Urea'].max())}
csv('metabolite_qc.csv',[{'feature':c,'zero_fraction':float((M[c]==0).mean()),'missing_fraction':float(M[c].isna().mean()),'minimum':float(M[c].min()),'maximum':float(M[c].max()),'retained_primary':c in y.columns} for c in M])
csv('microbiome_qc.csv',[{'feature':c,'prevalence':float((G[c]>0).mean()),'retained_primary':c in x.columns} for c in G])
csv('retention_by_feeding_group.csv',[{'diet':diet,'all_subjects':g.Subject.nunique(),'paired_subjects':int(pairs.diet.eq(diet).sum())} for diet,g in meta.groupby('diet')])
seed=int(config['seed'])+len(x); signs,_=generate_signs(len(x),max_random=255,seed=seed)
results=[]
def evaluate(label,a,b,sg=None):
    if sg is None: sg,_=generate_signs(len(a),max_random=255,seed=int(config['seed'])+len(a))
    r=analyze_multiview({'Microbiome':a,'Metabolome':b},signs=sg,k_grid=tuple(config['k_grid']))
    row={'analysis':label,'n':len(a),'microbiome_features':a.shape[1],'metabolome_features':b.shape[1],'microbiome_p':r.block_results['Microbiome'].block_pvalue,'metabolome_p':r.block_results['Metabolome'].block_pvalue,'joint_p':r.joint_pvalue,'patterns':len(sg)}
    results.append(row); return row
primary=evaluate('primary_reproduction',x,y,signs)
original=json.loads((BASE/'primary_results.json').read_text()); assert primary['joint_p']==original['joint_pvalue']
for diet,g in pairs.groupby('diet',sort=True):
    a,b,_=_paired_changes(micro,metab,g)
    evaluate('feeding_stratum_'+diet,a,b)
loo=[]
for subject in x.index:
    r=evaluate('omit_'+subject,x.drop(index=subject),y.drop(index=subject))
    loo.append(r)
print('Completed 57 individual deletions',flush=True)
rng=np.random.default_rng(int(config['seed'])+40000)
sub=[]
for i in range(30):
    a=x.iloc[:,np.sort(rng.choice(x.shape[1],size=int(np.ceil(.8*x.shape[1])),replace=False))]
    b=y.iloc[:,np.sort(rng.choice(y.shape[1],size=int(np.ceil(.8*y.shape[1])),replace=False))]
    sub.append(evaluate('feature_subsample_'+str(i),a,b,signs))
print('Completed 30 feature subsamples',flush=True)
def differences(frame):
    a=frame.loc[pairs.pre_sample].set_axis(x.index); b=frame.loc[pairs.post_sample].set_axis(x.index)
    d=b-a; return d.loc[:,d.std(ddof=1)>1e-12]
ret=M.loc[:,y.columns]
variants={'raw_concentration':ret,'asinh_concentration':np.arcsinh(ret)}
positive=ret.where(ret>0).min(axis=0)/2
variants['log_half_min_positive_per_feature']=np.log2(ret.add(positive,axis=1))
relative=ret.div(ret.sum(axis=1),axis=0)
variants['log_total_relative_half_min']=np.log2(relative.add(relative.where(relative>0).min()/2,axis=1))
for label,frame in variants.items(): evaluate('metabolite_'+label,x,differences(frame),signs)
annotation=pd.read_csv(raw/'mtb.map.tsv',sep='\t')
high=set(annotation.loc[annotation['High.Confidence.Annotation'].astype(str).str.lower().eq('true'),'Compound'])
for label,cols in [('high_confidence',[c for c in y if c in high]),('detected_25pct',[c for c in y if (M[c]>0).mean()>=.25]),('exclude_urea',[c for c in y if c!='Urea'])]:
    evaluate('metabolite_'+label,x,y.loc[:,cols],signs)
for prevalence in [.10,.50]:
    a,b,_=_paired_changes(micro,metab,pairs,prevalence=prevalence)
    evaluate('microbiome_prevalence_'+str(prevalence),a,b,signs)
filtered=G.loc[:,x.columns]
minpositive=float(filtered.where(filtered>0).min().min())
for multiplier in [.1,1.]:
    a=differences(clr_transform(filtered,pseudocount=minpositive*multiplier))
    evaluate('microbiome_pseudocount_min_times_'+str(multiplier),a,y,signs)
cols=[c for c in filtered if c.lower()!='unclassified']
evaluate('microbiome_remove_unclassified',differences(clr_transform(filtered[cols])),y,signs)
urea_pairs=[r.subject_id for r in pairs.itertuples() if metab.loc[r.pre_sample,'Urea']>0 or metab.loc[r.post_sample,'Urea']>0]
quality['urea_positive_pair_n']=len(urea_pairs)
if urea_pairs:
    clean=pairs.loc[~pairs.subject_id.isin(urea_pairs)]
    if len(clean)>=6:
        a,b,_=_paired_changes(micro,metab,clean); evaluate('exclude_urea_positive_pairs',a,b)
sg,_=generate_signs(len(x),max_random=4095,seed=seed)
evaluate('larger_closed_subgroup_4096',x,y,sg)
print('Completed transformations, strata and 4096-pattern check',flush=True)
skewrows=[]
for name,frame in [('Microbiome',x),('Metabolome',y)]:
    for c in frame:
        v=frame[c].to_numpy(); centered=v-v.mean()
        skewrows.append({'view':name,'feature':c,'centered_sample_skewness':float(skew(centered,bias=False)),'direction_consistency':float(max((v>0).sum(),(v<0).sum())/len(v))})
csv('descriptive_skewness.csv',skewrows)
csv('all_sensitivity_results.csv',results); csv('leave_one_subject.csv',loo); csv('feature_subsampling.csv',sub)
save('quality_audit.json',quality)
summary={'primary':primary,'quality':quality,'leave_one_subject':{'significant':sum(r['joint_p']<=.05 for r in loo),'total':len(loo),'max_joint_p':max(r['joint_p'] for r in loo)},'feature_subsamples':{'significant':sum(r['joint_p']<=.05 for r in sub),'total':len(sub),'max_joint_p':max(r['joint_p'] for r in sub)},'other_sensitivities':[r for r in results if not r['analysis'].startswith(('omit_','feature_subsample_'))],'skewness_descriptive_only':{name:{'median_absolute':float(np.median([abs(r['centered_sample_skewness']) for r in skewrows if r['view']==name])),'max_absolute':float(max(abs(r['centered_sample_skewness']) for r in skewrows if r['view']==name))} for name in ['Microbiome','Metabolome']}}
save('summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
