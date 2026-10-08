#!/usr/bin/env python3
"""Copy existing MX outputs for review. No science execution or metric edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selected-config',type=Path,required=True)
    for stage in ('e0','e1','e2','e3'):
        p.add_argument('--'+stage,type=Path)
    a=p.parse_args(); out=a.output.resolve()
    if out.exists(): p.error('use a new review output directory')
    if not a.selected_config.is_file(): p.error('selected config does not exist')
    out.mkdir(parents=True)
    shutil.copyfile(a.selected_config,out/'selected_config.json')
    for stage in ('e0','e1','e2','e3'):
        source=getattr(a,stage)
        if source is None:
            continue  # Reducer will mark missing evidence, not invent a result.
        source=source.resolve(); label=stage.upper()
        if not source.is_dir() or not (source/(label+'.json')).is_file():
            p.error(label+' requires existing stage directory and stage JSON')
        if out.is_relative_to(source):
            p.error('output must not be inside an input stage directory')
        for entry in source.rglob('*'):
            if entry.is_symlink() or not entry.resolve().is_relative_to(source):
                p.error('stage symlinks/external paths are forbidden')
        dest=out/'raw'/label
        shutil.copytree(source,dest)
        obj=json.loads((dest/(label+'.json')).read_text())
        records=[]
        for item in obj.get('raw_files',[]):
            path=(source/item['path']).resolve()
            if not path.is_relative_to(source) or not path.is_file() or sha(path)!=item['sha256']:
                p.error(label+' raw file missing/unsafe/hash differs')
            record=dict(item)
            record['path']=(Path('raw')/label/path.relative_to(source)).as_posix()
            records.append(record)
        for filename,kind in [('GUARD.json','guard'),('RUN_STATUS.json','run_status')]:
            path=dest/filename
            if not path.is_file():
                p.error(label+' launcher evidence missing: '+filename)
            relative=(Path('raw')/label/filename).as_posix()
            records=[x for x in records if x.get('kind')!=kind]
            records.append({'path':relative,'sha256':sha(path),'kind':kind})
        obj['raw_files']=records
        obj['guard_sha256']=sha(dest/'GUARD.json')
        (out/(label+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'review_dir':str(out),'note':'raw results copied; missing stages remain inconclusive'}))


if __name__=='__main__':
    main()
