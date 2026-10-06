"""Copy explicitly supplied, already existing artifacts after checksum validation.

Operator CLI only; no model/path upload is exposed by the Streamlit app.
"""
import argparse
import json
import shutil
from pathlib import Path
from hifimobinet.registry import manifest,repository_root,sha256,within


def copy_verified(source,target,expected,size):
    if not source.is_file(): raise FileNotFoundError(f'Missing source: {source.name}')
    if source.stat().st_size!=size or sha256(source)!=expected:
        raise ValueError(f'Checksum/size differs from manifest: {source.name}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==size and sha256(target)==expected:return 'already_verified'
        raise ValueError(f'Refusing to overwrite a different file: {target.name}')
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    if sha256(target)!=expected:raise ValueError('Copied file failed verification')
    return 'copied_and_verified'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q05-root',type=Path,help='Authorized existing Q05 archive root')
    p.add_argument('--harvard-root',type=Path,help='Authorized directory containing harvard_baseline, harvard_mrf, harvard_seq, harvard_piper')
    a=p.parse_args(argv);root=repository_root();records=[]
    if not a.q05_root and not a.harvard_root:p.error('Provide an existing authorized artifact directory')
    try:
        if a.q05_root:
            for m in manifest(root)['models']:
                item=m['artifact'];relative=m['historical_label']+'/'+item['filename']
                src=within(a.q05_root.resolve(),relative);dst=root/'models/weights'/item['filename']
                result=copy_verified(src,dst,item['sha256'],item['bytes'])
                records.append({'model_id':m['id'],'source':str(src),'destination':str(dst),'sha256':item['sha256'],'status':result})
        if a.harvard_root:
            catalog=json.loads((root/'demo/audio_manifest.json').read_text(encoding='utf-8'))
            for s in catalog['samples']:
                src=within(a.harvard_root.resolve(),'harvard_'+s['relative_path']);dst=root/'.local/audio'/s['relative_path']
                result=copy_verified(src,dst,s['sha256'],s['bytes'])
                records.append({'model_id':s['model_id'],'source':str(src),'destination':str(dst),'sha256':s['sha256'],'status':result})
    except (OSError,ValueError) as error:p.exit(2,f'Asset installation stopped: {error}\n')
    (root/'.local').mkdir(exist_ok=True)
    (root/'.local/assets-import.json').write_text(json.dumps({'classification':'local_copy_only','records':records},indent=2),encoding='utf-8')
    print(f'{len(records)} existing artifacts verified. No training, export, quantization or inference occurred.')


if __name__=='__main__':main()
