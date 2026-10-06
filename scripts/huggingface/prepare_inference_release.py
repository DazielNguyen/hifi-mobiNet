"""Prepare an allowlisted inference-only candidate; no network or model execution."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from hifimobinet.artifacts import safe_path
from hifimobinet.registry import sha256

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = 'DazielNguyen/hifi-mobiNet-inference'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    destination = args.destination.absolute()
    if destination.exists() or any(part.is_symlink() for part in [destination, *destination.parents]):
        raise ValueError('Use a new destination without symlink components')
    models = json.loads((ROOT/'models/manifest.json').read_text())['models']
    plan = []
    for model in models:
        record = model['artifact']
        name = 'onnx-q05/' + record['filename']
        source = safe_path(args.assets, name)
        if source.stat().st_size != record['bytes'] or sha256(source) != record['sha256']:
            raise ValueError('Source ONNX identity differs from the selected manifest')
        plan.append((source, name))
        config_name = 'configs/' + model['id'] + '.json'
        plan.append((safe_path(args.assets, config_name), config_name))
    plan.extend([
        (ROOT/'LICENSE', 'LICENSE'),
        (ROOT/'licenses/Piper-MIT.txt', 'licenses/Piper-MIT.txt'),
        (ROOT/'deploy/huggingface/inference-card.md', 'README.md'),
        (ROOT/'deploy/huggingface/inference-notices.md', 'INFERENCE-NOTICES.md'),
    ])
    for source, name in plan:
        if not source.is_file() or source.is_symlink():
            raise ValueError('Only existing regular allowlisted files are accepted')
    destination.mkdir(parents=True)
    for source, name in plan:
        output = safe_path(destination, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        if sha256(output) != sha256(source):
            raise ValueError('Copied file differs from source')
    artifacts = {'schema_version':1, 'repo_id':REPO_ID, 'precision':'FP32',
                 'format':'ONNX', 'sample_rate':22050,
                 'classification':'existing_Q05_engineering_artifacts_not_new_exports',
                 'excluded':['training checkpoints','training logs','optimizer state','Harvard text/audio','native frontend binaries'],
                 'models':[{'id':m['id'], 'artifact':{key:m['artifact'][key] for key in ('filename','sha256','bytes','format','precision','opset')},
                            'license':'MIT_team_owned_weights_only' if m['id']!='piper-original' else 'upstream_MIT_with_retained_attribution'} for m in models]}
    (destination/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    (destination/'SHA256SUMS').write_text(''.join(
        m['artifact']['sha256']+'  onnx-q05/'+m['artifact']['filename']+'\n' for m in models))
    rows=[{'relative_path':f.relative_to(destination).as_posix(),'bytes':f.stat().st_size,'sha256':sha256(f)}
          for f in sorted(destination.rglob('*')) if f.is_file()]
    inventory={'schema_version':1,'repo_id':REPO_ID,'scope':'Inference-only package; excludes self to avoid circular hashes.',
               'files':rows,'public_release_authority':'Author chose a separate public inference repository while keeping original checkpoints private.'}
    (destination/'release-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'repo_id':REPO_ID,'files':len(rows)+1,'bytes_excluding_inventory':sum(r['bytes'] for r in rows)}))


if __name__ == '__main__':
    main()
