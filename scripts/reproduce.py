"""Convenience entry point; each stage can also be run independently."""
import argparse,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['check','public','simulations','rats','tables','figures'])
    p.add_argument('--quick',action='store_true',help='Reduced simulations written to results/quick')
    p.add_argument('--inputs',type=Path,help='Rat input paths JSON')
    a=p.parse_args()
    if a.quick and a.stage!='simulations':p.error('--quick is available only for simulations')
    stages={
      'check':[['-m','pytest','-q'],['scripts/09_run_method_validation.py','--checks-only'],['examples/run_example.py']],
      'public':[['scripts/download_public_data.py'],['scripts/01_prepare_inputs.py','--scope','public'],['scripts/03_run_public_validation.py','--sensitivity'],['scripts/run_he_validation.py'],['scripts/run_he_sensitivity.py']],
      'simulations':[['scripts/02_run_simulation.py'],['scripts/09_run_method_validation.py','--simulations-only'],['scripts/11_run_structured_correlation_validation.py'],['scripts/13_run_asymmetry_gradient.py'],['scripts/18_fixed_scale_ablation.py']],
      'rats':[['scripts/01_prepare_inputs.py','--scope','local']+(['--inputs',str(a.inputs)] if a.inputs else []),['scripts/04_run_local_application.py'],['scripts/15_run_ko_clr_sensitivity.py']],
      'tables':[['scripts/06_prepare_figure_data.py','--section','all','--refresh-expensive'],['scripts/assemble_figure_data.py']],
      'figures':[['scripts/build_figures.py','--scope','public']]}
    if a.inputs:os.environ["AS_MVSPD_LOCAL_INPUTS"]=str(a.inputs.resolve())
    commands=stages[a.stage]
    if a.quick:
        commands=commands[:2]
        for command in commands:command.append('--quick')
    for command in commands:
        print('RUN',sys.executable,*command,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
if __name__=='__main__':main()
