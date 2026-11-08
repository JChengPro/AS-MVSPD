"""Assemble a complete recomputed figure-input directory; fail if a table is missing."""
import argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/recomputed_figure_data')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/publication_figures/source_data'
    required={f.name for f in (ROOT/'figures/source_data').glob('*.csv') if f.name not in ['HE_published_baselines.csv']}
    missing=[]
    for name in sorted(required):
        if name=='fixed_scale_ablation_summary.csv':src=ROOT/'results/method_validation/fixed_scale_ablation/summary.csv'
        elif name.startswith('public_') or name=='Table_1_public_validation.csv':src=ROOT/'results/public_validation'/name
        elif name=='real_data_published_baselines.csv':src=ROOT/'results/method_validation'/name
        else:src=source/name
        if not src.exists():missing.append(str(src.relative_to(ROOT)))
        else:shutil.copy2(src,a.output/name)
    # Rat figure-source tables are included only when that workflow was run.
    for pattern in ['local_*.csv','qc_*.csv']:
        for f in source.glob(pattern):shutil.copy2(f,a.output/f.name)
    if missing:raise FileNotFoundError('Run the corresponding full stages first. Missing tables:\n'+'\n'.join(missing))
    print('Recomputed figure inputs:',a.output)
if __name__=='__main__':main()
