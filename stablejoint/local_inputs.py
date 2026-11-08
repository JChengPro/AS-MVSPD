"""User-configured rat input paths, with no dependency on the original workspace."""
import json
import os
from pathlib import Path
from .local_preprocessing import Paths


def configured_paths(project: Path, config: Path | None = None) -> Paths:
    filename = config or Path(os.environ.get('AS_MVSPD_LOCAL_INPUTS', project/'.private/local_input_paths.json'))
    if not filename.is_file():
        raise FileNotFoundError('Provide a local-inputs JSON using --inputs or AS_MVSPD_LOCAL_INPUTS. See config/local_inputs.example.json.')
    raw = json.loads(filename.read_text())
    files = {k:Path(v).expanduser() for k,v in raw.items() if k in {'sample_mapping','species','ko','metabolite','ko_annotation'}}
    files = {k:p if p.is_absolute() else filename.parent/p for k,p in files.items()}
    for name in ['sample_mapping','species','ko','metabolite']:
        if name not in files or not files[name].is_file():raise FileNotFoundError(f'Missing {name}: {files.get(name)}')
    return Paths(project_root=project,analysis_root=project,results=project/'results',sample_map=files['sample_mapping'],
                 species=files['species'],ko=files['ko'],metabolite=files['metabolite'],taxfun=project/'data/raw/unused_taxfun.txt')
