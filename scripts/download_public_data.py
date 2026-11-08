"""Download the exact public input versions, and verify SHA-256 before use."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true',help='Verify locally present inputs without network access')
    args=parser.parse_args()
    records=json.loads((ROOT/'config/public_sources.json').read_text())
    for row in records:
        p=ROOT/'data/raw'/row['path']
        if p.exists():
            content=p.read_bytes()
        elif args.verify_only:
            raise FileNotFoundError(p)
        else:
            with urlopen(Request(row['url'],headers={'User-Agent':'AS-MVSPD/0.1.0'}),timeout=60) as response:content=response.read()
        if hashlib.sha256(content).hexdigest()!=row['sha256']:
            raise ValueError(f'Input checksum mismatch: {row["path"]}. No replacement accepted.')
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        print('verified',row['path'])

if __name__=='__main__':main()
