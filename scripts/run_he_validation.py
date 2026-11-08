"""HE primary test and 20 additional subgroup seeds used in the HE checks."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stablejoint import analyze_multiview,generate_signs
from stablejoint.public_datasets import load_pair


def main():
    config=json.loads((ROOT/'config/experiment.json').read_text());out=ROOT/'results/he_validation';out.mkdir(parents=True,exist_ok=True)
    x,y,pairs=load_pair(ROOT/'data/public_prepared/HE_INFANTS_MFGM_2019');seed=int(config['seed'])+len(x)
    blocks={'Microbiome':x,'Metabolome':y};signs,full=generate_signs(len(x),max_random=255,seed=seed)
    r=analyze_multiview(blocks,signs=signs,k_grid=tuple(config['k_grid']));np.save(out/'primary_signs.npy',signs)
    summary={'dataset':'HE_INFANTS_MFGM_2019','subjects':len(x),'joint_pvalue':r.joint_pvalue,'primary_seed':seed,'sign_patterns':len(signs),'full_group':full,
             'views':[{'view':k,'features':blocks[k].shape[1],'pvalue':b.block_pvalue,'selected_k':b.selected_k} for k,b in r.block_results.items()]}
    (out/'primary_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    for k,b in r.block_results.items():b.feature_table.to_csv(out/(k.lower()+'_feature_results.csv'),index=False)
    rows=[]
    for offset in range(1,21):
        sg,_=generate_signs(len(x),max_random=255,seed=seed+offset);rr=analyze_multiview(blocks,signs=sg,k_grid=tuple(config['k_grid']))
        rows.append(dict(seed=seed+offset,microbiome_p=rr.block_results['Microbiome'].block_pvalue,metabolome_p=rr.block_results['Metabolome'].block_pvalue,joint_p=rr.joint_pvalue))
    pd.DataFrame(rows).to_csv(out/'seed_sensitivity.csv',index=False)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
