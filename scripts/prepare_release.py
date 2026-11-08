"""Build an explicit code-upload inventory and a local preparation ZIP."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def selected_files():
    names=['README.md','pyproject.toml','requirements.txt','pytest.ini','CITATION.cff','.gitignore','LICENSE','LICENSE.md','LICENSE.txt']
    files=[ROOT/n for n in names if (ROOT/n).is_file()]
    for folder in ['stablejoint','scripts','config','tests','examples','docs','reference','figures/source_data','figures/renderers']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix in ['.pyc','.pyo','.docx','.xlsx']:continue
            if p.name in ['UPLOAD_MANIFEST.json','release_readiness.json']:continue
            files.append(p)
    files.append(ROOT/'figures/legends.json')
    return sorted(set(files))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT.parent/'AS-MVSPD_code_preparation.zip')
    a=p.parse_args()
    files=selected_files()
    manifest=[{'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
    pending=[]
    if not any((ROOT/n).is_file() for n in ['LICENSE','LICENSE.md','LICENSE.txt']):pending.append('Author/rights-holder license decision and final LICENSE')
    pending.extend(['Confirm rat study input and individual-level plotting data access arrangements',
                    'Create and verify the intended remote GitHub repository',
                    'Archive submitted software version and test data; record the actual permanent URL/DOI'])
    readiness={'stage':'local_preparation','uploaded':False,'submission_ready':False,'pending':pending,'selected_payload_files':len(files),
               'excluded':['.private/','data/','results/','verification/','generated figures','Word manuscripts','Excel workbooks'],
               'note':'The inventory hashes cover payload files; manifest/readiness metadata are included in the ZIP separately.'}
    (ROOT/'docs/UPLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'docs/release_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files+[ROOT/'docs/UPLOAD_MANIFEST.json',ROOT/'docs/release_readiness.json']:
            rel=f.relative_to(ROOT)
            if rel.parts[0] in ['data','.private','results','verification']:raise ValueError(f'Excluded file in inventory: {rel}')
            z.write(f,'AS-MVSPD/'+str(rel))
    print(json.dumps({'zip':str(a.output),'payload_files':len(files),'bytes':a.output.stat().st_size,'pending':pending},indent=2))
if __name__=='__main__':main()
