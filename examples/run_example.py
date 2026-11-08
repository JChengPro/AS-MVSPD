"""Run the bundled simulated example with all 256 individual sign patterns."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stablejoint.cli import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'--view',f'Microbiome={ROOT}/examples/microbiome_paired_changes.csv','--view',f'Metabolome={ROOT}/examples/metabolome_paired_changes.csv','--output',str(ROOT/'results/example')]
    main()
