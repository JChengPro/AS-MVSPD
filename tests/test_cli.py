import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def test_cli_reproduces_bundled_example(tmp_path):
    run=subprocess.run([sys.executable,'-m','stablejoint','--view',f'A={ROOT}/examples/microbiome_paired_changes.csv','--view',f'B={ROOT}/examples/metabolome_paired_changes.csv','--output',str(tmp_path/'result')],cwd=ROOT,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    result=json.loads((tmp_path/'result/summary.json').read_text())
    expected=json.loads((ROOT/'examples/expected_output.json').read_text())
    assert result['joint_pvalue']==expected['joint_pvalue']
    assert result['sign_patterns']==256 and result['complete_full_group']


def test_cli_rejects_mismatched_subject_order(tmp_path):
    frame=pd.read_csv(ROOT/'examples/metabolome_paired_changes.csv',index_col=0)
    frame.iloc[::-1].to_csv(tmp_path/'reversed.csv')
    run=subprocess.run([sys.executable,'-m','stablejoint','--view',f'A={ROOT}/examples/microbiome_paired_changes.csv','--view',f'B={tmp_path}/reversed.csv','--output',str(tmp_path/'result')],cwd=ROOT,text=True,capture_output=True)
    assert run.returncode!=0
    assert 'same subject order' in run.stderr
    assert not (tmp_path/'result').exists()
