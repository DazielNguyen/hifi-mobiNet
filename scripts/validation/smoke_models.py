"""One short functional utterance per installed manifest model; NOT a benchmark."""
import argparse
import datetime
import importlib.metadata
import io
import json
import wave
from pathlib import Path
from hifimobinet.inference import Voice
from hifimobinet.registry import manifest,repository_root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enable-verified-models',action='store_true',help='Update checked-in model status only for models whose synthesis passes')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/functional-smoke'))
    args=p.parse_args(argv);root=repository_root();data=manifest(root);records=[]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for m in data['models']:
        result={'model_id':m['id'],'artifact_sha256':m['artifact']['sha256'],'classification':'new_validation','tts_passed':False}
        try:
            wav=Voice(m['id'],root).synthesize('The birch canoe slid on the smooth planks.')
            with wave.open(io.BytesIO(wav),'rb') as w:
                assert w.getnchannels()==1 and w.getframerate()==22050 and w.getnframes()>0
                result.update(tts_passed=True,frames=w.getnframes(),sample_rate=w.getframerate())
            with (args.output_dir/(m['id']+'.wav')).open('xb') as out:out.write(wav)
        except Exception as error:
            result['error_type']=type(error).__name__
            result['tts_passed']=False
        if args.enable_verified_models:
            m['local_functional_validation']={'tts_passed':result['tts_passed'],'record':'docs/functional-validation.json'}
        records.append(result)
    report={'classification':'new_validation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'onnxruntime':importlib.metadata.version('onnxruntime'),'provider':'CPUExecutionProvider',
            'intra_op_threads':2,'inter_op_threads':1,'is_benchmark':False,'records':records}
    path=root/'docs/functional-validation.json' if args.enable_verified_models else args.output_dir/'report.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if args.enable_verified_models:(root/'models/manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    passed=sum(r['tts_passed'] for r in records);print(f'Functional synthesis passed: {passed}/{len(records)}. No benchmark metrics collected.')
    return 0 if passed==len(records) else 2


if __name__=='__main__':raise SystemExit(main())
