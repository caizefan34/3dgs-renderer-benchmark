#!/usr/bin/env python3
"""Archive an existing result directory without executing scientific code."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('review_dir',type=Path)
    p.add_argument('output_zip',type=Path)
    a=p.parse_args()
    root=a.review_dir.resolve(); output=a.output_zip.resolve()
    if not root.is_dir() or output.is_relative_to(root) or output.exists():
        p.error('Existing review directory and a new ZIP outside it are required')
    if not (root/'VERDICT.json').is_file() or not (root/'BES0_REPORT.md').is_file():
        p.error('Provide machine-generated verdict and report first, including for blocked runs')
    files=sorted(x for x in root.rglob('*') if x.is_file() and x.name!='RESULT_MANIFEST.sha256')
    if any(x.is_symlink() or not x.resolve().is_relative_to(root) for x in files):
        p.error('Symlinks/external paths are not allowed in result archive')
    lines=[hashlib.sha256(x.read_bytes()).hexdigest()+'  '+x.relative_to(root).as_posix() for x in files]
    manifest=root/'RESULT_MANIFEST.sha256'
    manifest.write_text('\n'.join(lines)+'\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for x in files+[manifest]:
            z.write(x,root.name+'/'+x.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP CRC failed')
    print(json.dumps({'file':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__=='__main__':
    main()
