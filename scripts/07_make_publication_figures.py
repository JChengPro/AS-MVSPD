"""Compatibility entry point for the current manuscript figure builder."""
from pathlib import Path
import runpy
if __name__=='__main__':runpy.run_path(str(Path(__file__).with_name('build_figures.py')),run_name='__main__')
