"""Command-line entry point for aligned paired-difference matrices."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import analyze_multiview,generate_signs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',action='append',required=True,metavar='NAME=FILE.csv',help='Repeat for at least two views; first column is subject ID')
    p.add_argument('--k-grid',type=int,nargs='+',default=[1,2,5,10,20])
    p.add_argument('--max-random',type=int,default=4095,help='Maximum non-identity sign patterns; a closed subgroup is generated')
    p.add_argument('--seed',type=int,default=20260803)
    p.add_argument('--output',type=Path,default=Path('results/example'))
    args=p.parse_args();blocks={}
    for entry in args.view:
        name,sep,path=entry.partition('=')
        if not sep or not name or name in blocks:p.error('Each view needs a unique NAME=FILE.csv')
        df=pd.read_csv(path,index_col=0);df.index=df.index.astype(str)
        if not df.index.is_unique or not df.columns.is_unique or len(df)<2 or df.shape[1]<1:p.error('Matrices need unique identifiers, >=2 subjects and >=1 feature')
        if not np.isfinite(df.to_numpy(dtype=float)).all():p.error('Matrices must contain finite numeric differences')
        blocks[name]=df
    if len(blocks)<2:p.error('At least two views required')
    if not args.k_grid or min(args.k_grid)<1:p.error('k-grid values must be positive')
    first=next(iter(blocks.values()))
    if any(not first.index.equals(b.index) for b in blocks.values()):p.error('All matrices must use the same subject order')
    signs,full=generate_signs(len(first),max_random=args.max_random,seed=args.seed)
    r=analyze_multiview(blocks,signs=signs,k_grid=args.k_grid)
    args.output.mkdir(parents=True,exist_ok=True)
    summary={'joint_pvalue':r.joint_pvalue,'view_pvalues':{k:b.block_pvalue for k,b in r.block_results.items()},'selected_k':{k:b.selected_k for k,b in r.block_results.items()},'sign_patterns':len(signs),'complete_full_group':full,'seed':args.seed,'k_grid':args.k_grid}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for k,b in r.block_results.items():
        # Filenames use ordinal IDs so view names cannot escape the output directory.
        i=list(blocks).index(k)+1;b.feature_table.to_csv(args.output/f'view_{i}_feature_results.csv',index=False)
    print(json.dumps(summary,indent=2))
