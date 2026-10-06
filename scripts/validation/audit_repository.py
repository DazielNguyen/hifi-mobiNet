"""Audit tracked content and every reachable local Git revision before handoff.

This is a bounded heuristic secret/content scan, not a security guarantee.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXCLUDED_METADATA={'docs/release-manifest.json','docs/final-audit.json'}
BAD_SUFFIXES={'.ckpt','.pt','.pth','.onnx','.safetensors','.dll','.pyd','.so','.zip'}
PATTERNS=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
          re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'AKIA[0-9A-Z]{16}'),
          re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{35,}'),
          re.compile(rb'(?:C:[/\\]+Users[/\\]|'+b'/mnt/'+b'c/Users/|'+b'/home/'+b'capstone/'+b')')]


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def digest(data):return hashlib.sha256(data).hexdigest()


def inspect(name,data):
    if Path(name).suffix.lower() in BAD_SUFFIXES:raise ValueError('Excluded artifact in Git: '+name)
    if len(data)>10*1024*1024:raise ValueError('Unexpected large Git file: '+name)
    if any(x in Path(name).parts for x in ('.venv','.local','__pycache__','node_modules')):
        raise ValueError('Excluded environment/cache path in Git: '+name)
    if name=='.env' or name.endswith('/secrets.toml'):raise ValueError('Secret configuration file in Git')
    for pattern in PATTERNS:
        if pattern.search(data):raise ValueError('Sensitive-pattern review required: '+name)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-manifest',action='store_true')
    p.add_argument('--record',action='store_true')
    args=p.parse_args(argv)
    try:
        source_map=json.loads((ROOT/'docs/source-map.json').read_text(encoding='utf-8'))['entries']
        mappings={entry['destination']:entry for entry in source_map}
        for entry in source_map:
            file=ROOT/entry['destination']
            if digest(file.read_bytes())!=entry['destination_sha256']:
                raise ValueError('Source-map destination hash is stale: '+entry['destination'])
        names=git('ls-files','-z').decode().rstrip('\0').split('\0')
        records=[]
        for name in names:
            path=ROOT/name
            data=path.read_bytes();inspect(name,data)
            if name in EXCLUDED_METADATA:continue
            src=mappings.get(name)
            records.append({'file':name,'sha256':digest(data),'bytes':len(data),
                            'source':src['source'] if src else 'new_local_repository_work',
                            'source_sha256':src['source_sha256'] if src else None,
                            'source_commit':src['source_commit'] if src else None,
                            'classification':src['classification'] if src else 'new_repository_implementation_or_documentation',
                            'rights':src['license_status'] if src else 'author_license_decision_pending',
                            'release_status':'local_only_pending_author_release_review'})
        # Scan every unique historical blob, not merely the current checkout.
        objects=git('rev-list','--objects','--all').decode().splitlines()
        seen=set();blobs=0
        for line in objects:
            parts=line.split(' ',1)
            if len(parts)!=2:continue
            oid,name=parts
            if oid in seen:continue
            seen.add(oid)
            if git('cat-file','-t',oid).strip()!=b'blob':continue
            inspect(name,git('cat-file','blob',oid));blobs+=1
        if git('remote').strip():raise ValueError('Unexpected Git remote configured')
        if args.write_manifest:
            models=json.loads((ROOT/'models/manifest.json').read_text())['models']
            out={'schema_version':1,'scope':'Tracked working files; self and final-audit excluded to avoid circular hashes.',
                 'excluded_self_metadata':sorted(EXCLUDED_METADATA),'files':records,
                 'models':[{'id':m['id'],'checkpoint':m['checkpoint'],'artifact':m['artifact'],'release_status':m['release_status']} for m in models]}
            (ROOT/'docs/release-manifest.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
        else:
            saved=json.loads((ROOT/'docs/release-manifest.json').read_text(encoding='utf-8'))
            expected={e['file']:e['sha256'] for e in saved['files']}
            actual={e['file']:e['sha256'] for e in records}
            if actual!=expected:raise ValueError('Release manifest differs from tracked working files; inspect changes before refreshing it')
        report={'classification':'new_validation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'checked_head':git('rev-parse','HEAD').decode().strip(),'tracked_files':len(names),'unique_history_blobs':blobs,
                'source_map_entries':len(source_map),'remote_count':0,'status':'passed',
                'limits':'Heuristic content scan only; no guarantee of complete secret detection or license clearance. Source repositories verified separately.'}
        if args.record:(ROOT/'docs/final-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as error:
        p.exit(2,f'Audit failed: {error}\n')


if __name__=='__main__':main()
