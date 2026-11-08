"""Recompute real-data comparators without rerunning simulation sections."""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('method_validation',ROOT/'scripts/09_run_method_validation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seed=int(json.loads((ROOT/'config/experiment.json').read_text())['seed'])
    table=module.run_real_data_baselines(seed)
    out=ROOT/'results/method_validation';out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'real_data_published_baselines.csv',index=False)
    print(table.to_string(index=False))
