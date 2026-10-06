"""Package an already verified staging directory into local ZIPs; never publish."""
import argparse
import json
from pathlib import Path
import zipfile
from hifimobinet.artifacts import Package, describe_file
from hifimobinet.registry import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staging',type=Path,required=True)
    args=p.parse_args();root=args.staging.resolve()
    verified=Package(root)
    try:
        data=json.loads((root/'manifests/artifact-manifest.json').read_text(encoding='utf-8'))
        rows={r['relative_path']:r for r in data['artifacts']}
        rows['manifests/artifact-manifest.json']=describe_file(root/'manifests/artifact-manifest.json','manifests/artifact-manifest.json')
        rows['SHA256SUMS']=describe_file(root/'SHA256SUMS','SHA256SUMS')
        common={n for n in rows if n.startswith('notices/') or n in ('README.md','configs/frontend.json')}
        groups={}
        for model,folder in [('baseline-resblock2','baseline'),('parallel-ir','parallel-ir'),('sequential-ir','sequential-ir'),('piper-original','piper')]:
            groups['checkpoint-'+folder]=common|{n for n,r in rows.items() if r.get('model_id')==model and r.get('artifact_type') in ('training_checkpoint','recovered_configuration','recovered_checkpoint_hparams','training_written_hparams')}
        groups['onnx-q05']=common|{n for n in rows if n.startswith(('onnx-q05/','configs/'))}
        groups['audio-harvard']=common|{n for n in rows if n.startswith('audio/')}|{'manifests/audio-manifest.json','manifests/harvard720.json'}
        groups['metadata']=set(rows)-{n for n in rows if n.startswith(('audio/','checkpoints/','onnx-q05/'))}
        out=root/'archives'
        out.mkdir(exist_ok=True)
        # No archive is overwritten, even if it appears to come from an earlier run.
        paths={g:out/f'hifi-mobiNet-v0.1.0-{g}.zip' for g in groups}
        if any(path.exists() for path in paths.values()):
            raise ValueError('Archive already exists; inspect it before choosing another output/preparation')
        result=[]
        for group,names in groups.items():
            package={'schema_version':1,'release_id':'v0.1.0-proposed-unpublished','package_id':group,
                     'classification':'local_transfer_only','artifacts':[rows[n] for n in sorted(names)]}
            if sum(rows[n]['bytes'] for n in names)>=2*1024**3:
                raise ValueError('Group is too large for the requested 2 GiB asset limit')
            path=paths[group]
            with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for name in sorted(names):z.write(root/name,name)
                z.writestr('manifests/packages/'+group+'.json',json.dumps(package,indent=2)+'\n')
            row={'filename':path.name,'bytes':path.stat().st_size,'sha256':sha256(path),'members':len(names)+1,'format':'ZIP_STORED','status':'local_unpublished'}
            if row['bytes']>=2*1024**3:raise ValueError('Archive is not under 2 GiB')
            result.append(row)
            print(json.dumps(row),flush=True)
        index={'release_id':'v0.1.0-proposed-unpublished','archives':result,'limits':'No upload or release. Checksums verify bytes, not distribution rights or scientific acceptance.'}
        with (out/'archive-index.json').open('x',encoding='utf-8') as f:f.write(json.dumps(index,indent=2)+'\n')
        with (out/'SHA256SUMS').open('x',encoding='utf-8') as f:
            for r in result:f.write(r['sha256']+'  '+r['filename']+'\n')
            f.write(sha256(out/'archive-index.json')+'  archive-index.json\n')
        # Reopen the finished archives and verify every member against its manifest.
        for path in paths.values():
            check=Package(path);check.close()
        print('All seven finished archives verified; no archive modified after hashing.',flush=True)
    finally:verified.close()


if __name__=='__main__':main()
