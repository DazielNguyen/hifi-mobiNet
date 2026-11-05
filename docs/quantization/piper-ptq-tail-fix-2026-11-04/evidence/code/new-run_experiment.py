"""Isolated PTQ experiments. Original source/graphs are read-only.

Quality screening on PC CPU, NOT an N100 timing result. Full 2^4 block factorial
plus finer scopes. All quantization uses the original 60 train-only calibration
rows. Diagnostic texts never enter calibration. No original artifact promotion.
"""
import os
os.environ['NNCF_TELEMETRY_DISABLED'] = '1'
os.environ['OPENVINO_TELEMETRY_DISABLED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['ONEDNN_MAX_CPU_ISA'] = 'AVX2'
os.environ['DNNL_MAX_CPU_ISA'] = 'AVX2'
import argparse, hashlib, importlib.util, json, platform, re, sys, time, traceback, wave, warnings
# Process-local import blocker; no usage telemetry or model metadata leaves this process.
sys.modules['openvino_telemetry'] = None
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import openvino as ov
import onnx

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
EVIDENCE = REPO/'results/edge-n100-openvino/2026-11-01/evidence'
ORIGINAL = EVIDENCE.parent/'.local/originals/final'
LEGACY = ORIGINAL/'work-n100-clean-20261101T075458Z'
MODELS = ['piper-seqir-no-vits2-cpn','piper-parir-no-vits2-cpn','piper-no-vits2-cpn']
SCALES = np.array([0.667,1.0,0.8],dtype=np.float32)
CONFIG = {'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':1,'INFERENCE_NUM_THREADS':4,
          'INFERENCE_PRECISION_HINT':'f32','EXECUTION_MODE_HINT':'ACCURACY'}
WEIGHTED = {'Convolution','GroupConvolution','ConvolutionBackpropData','GroupConvolutionBackpropData','MatMul'}

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

common=load_module('original_common',REPO/'scripts/evaluation/onnx/common.py')
old=load_module('original_n100_suite',EVIDENCE/'openvino_suite.py')
guard_module=load_module('original_numeric_guard',EVIDENCE/'diagnose_n100_calibration.py')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def utc(): return datetime.now(timezone.utc).isoformat()

def feeds(row,seed=None,names=None):
    ids=row['phoneme_ids']
    result={'input':np.array([ids],dtype=np.int64),'input_lengths':np.array([len(ids)],dtype=np.int64),'scales':SCALES.copy()}
    if names:
        rng=np.random.default_rng(seed)
        result[names['duration_input']]=rng.standard_normal((1,2,len(ids))).astype(np.float32)
        result[names['acoustic_input']]=rng.standard_normal((1,192,10000)).astype(np.float32)
    return result

def block(node):
    name=node.get_friendly_name()
    for prefix,group in [('/enc_p/','E'),('/dp/','D'),('/flow/','F'),('/dec/','G')]:
        if name.startswith(prefix): return group
    if node.get_type_name() in ('ConvolutionBackpropData','GroupConvolutionBackpropData'):
        # Verify decoder attribution through immediate consumers, not a name guess.
        for port in node.output(0).get_target_inputs():
            if port.get_node().get_friendly_name().startswith('/dec/ups.'):
                return 'G'
    return 'other'

def variants():
    rows=[{'id':'fp32','mask':0}]
    # Most informative exclusions and single blocks first; still all 16 combinations.
    for mask in [7,4,1,2,8,5,6,3,9,10,12,11,13,14,15]:
        rows.append({'id':'blocks_'+''.join(g for i,g in enumerate('EDFG') if mask&(1<<i)),'mask':mask})
    rows += [{'id':x,'fine':x} for x in ['encoder_no_prior','encoder_attention_only','flow_prepost_only','flow_wavenet_only','decoder_internal_only','decoder_pointwise_only','decoder_depthwise_only','decoder_upsampling_only','legacy_recipe_new']]
    return rows

def scope(model,variant):
    ops=list(model.get_ops()); allowed=[]
    groups={g:[] for g in ['E','D','F','G','other']}
    for n in ops:
        if n.get_type_name() not in WEIGHTED: continue
        groups[block(n)].append({'name':n.get_friendly_name(),'type':n.get_type_name()})
        if n.get_type_name()=='MatMul': continue
        name=n.get_friendly_name(); group=block(n); choose=False
        if 'mask' in variant and group in 'EDFG': choose=bool(variant['mask']&(1<<'EDFG'.index(group)))
        fine=variant.get('fine')
        if fine=='encoder_no_prior': choose=group=='E' and not name.startswith('/enc_p/proj/')
        if fine=='encoder_attention_only': choose=group=='E' and '/attn_layers.' in name
        if fine=='flow_prepost_only': choose=group=='F' and bool(re.search(r'/flows\.[0246]/(?:pre|post)/',name))
        if fine=='flow_wavenet_only': choose=group=='F' and '/enc/' in name
        if fine=='decoder_internal_only': choose=group=='G' and any(s in name for s in ['/resblocks.','/mb_blocks.'])
        if fine=='decoder_pointwise_only': choose=group=='G' and any(s in name for s in ['/pw_expand/','/pw_project/'])
        if fine=='decoder_depthwise_only': choose=group=='G' and '/dw/' in name
        if fine=='decoder_upsampling_only': choose=group=='G' and 'BackpropData' in n.get_type_name()
        if fine=='legacy_recipe_new': choose=group in ('E','D','F')
        if choose: allowed.append(name)
    if variant.get('fine')=='legacy_recipe_new':
        ignored=[n.get_friendly_name() for n in ops
                 if ('/dec/' in n.get_friendly_name() and n.get_type_name() not in ('Constant','Parameter','Result'))
                 or (n.get_type_name() in WEIGHTED and n.get_friendly_name() not in allowed)
                 or (old.is_masked_duration_spline(n.get_friendly_name()) and n.get_type_name() not in ('Constant','Parameter','Result'))]
    else:
        # Isolate Conv inputs: all other computational nodes explicitly excluded.
        ignored=[n.get_friendly_name() for n in ops if n.get_type_name() not in ('Constant','Parameter','Result') and n.get_friendly_name() not in allowed]
    return allowed,sorted(set(ignored)),groups

def save_wav(path,y):
    y=np.asarray(y).reshape(-1)
    if not y.size or not np.isfinite(y).all(): raise ValueError('Empty or nonfinite audio')
    if y.size>22050*60: raise ValueError('Audio >60 s; retained as failed case, not trimmed')
    peak=float(np.abs(y).max()); rms=float(np.sqrt(np.mean(y.astype(np.float64)**2)))
    pcm=np.clip(y*(32767/max(.01,peak)),-32767,32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050);f.writeframes(pcm.tobytes())
    return {'samples':int(y.size),'seconds':y.size/22050,'raw_peak':peak,'raw_rms':rms,'wav_sha256':sha(path)}

def run_one(args):
    run=BASE/args.run; out=run/args.model/args.variant
    out.mkdir(parents=True,exist_ok=False)
    v=next(x for x in variants() if x['id']==args.variant) if args.variant!='historical_int8' else {'id':args.variant}
    source=ORIGINAL/'models'/args.model/'fp32.onnx'
    manifest=read(EVIDENCE/'models.json')
    item=next(x for x in manifest['models'] if x['id']==args.model)
    assert sha(source)==item['fp32']['sha256']
    record={'status':'running','started_utc':utc(),'model':args.model,'variant':v,'source_sha256':sha(source),
            'environment':{'platform':platform.platform(),'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(),
                           'python':platform.python_version(),'openvino':ov.__version__,'numpy':np.__version__},
            'config':CONFIG,'not_n100_benchmark':True,'script_sha256':sha(__file__),'input_texts_sha256':sha(BASE/'texts.json')}
    try:
        core=ov.Core()
        native=args.variant=='historical_int8' or args.native
        names=None
        if args.variant=='historical_int8':
            model=core.read_model(str(LEGACY/args.model/'int8.xml'))
            record['legacy_xml_sha256']=sha(LEGACY/args.model/'int8.xml')
            record['legacy_bin_sha256']=sha(LEGACY/args.model/'int8.bin')
        elif native:
            model=core.read_model(str(source))
        else:
            original=onnx.load(str(source)); matched,names=common.matched_noise_graph(original,192)
            assert common.strip_matched_noise(matched).SerializeToString()==common.without_random_nodes(original).SerializeToString()
            onnx.checker.check_model(matched)
            matched_path=out/'matched-fp32.onnx';onnx.save(matched,str(matched_path))
            model=core.read_model(str(matched_path))
        record['rng']='native graph RNG; not matched' if native else 'explicit paired duration/acoustic noise; same per-text seed across variants'
        record['noise_inputs']=names
        if args.variant not in ['fp32','historical_int8']:
            import nncf
            record['environment']['nncf']=nncf.__version__
            allowed,ignored,groups=scope(model,v)
            write(out/'scope.json',{'allowed':allowed,'ignored':ignored,'weighted_groups':groups})
            if not allowed:
                record['status']='not_applicable_no_matching_layers';return
            cal=[json.loads(s) for s in (EVIDENCE/'metadata/calibration60.candidate.jsonl').read_text().splitlines() if s.strip()]
            subset=[json.loads(s) for s in (EVIDENCE/'metadata/calibration_dataset_rows.jsonl').read_text().splitlines() if s.strip()]
            split=read(EVIDENCE/manifest['split']['file'])
            check=common.check_calibration(cal,split,subset);assert check['passed'] and check['n']==60
            record['calibration']=check
            prepared=[feeds(row,1234+i,names) for i,row in enumerate(cal)]
            dataset=nncf.Dataset(prepared)
            guard=guard_module.Recorder(out,np)
            t=time.perf_counter()
            model=guard.guarded_call(nncf.quantize,model,dataset,subset_size=60,preset=nncf.QuantizationPreset.MIXED,
                                    target_device=nncf.TargetDevice.CPU,fast_bias_correction=True,
                                    ignored_scope=nncf.IgnoredScope(names=ignored))
            record['quantization_s']=time.perf_counter()-t
            record['allowed_count']=len(allowed)
        ov.save_model(model,str(out/'model.xml'),compress_to_fp16=False)
        t=time.perf_counter();compiled=core.compile_model(model,'CPU',CONFIG);record['compile_s']=time.perf_counter()-t
        request=compiled.create_infer_request();rows=[]
        texts=read(BASE/'texts.json')
        if args.cases: texts=[x for x in texts if x['id'] in args.cases.split(',')]
        for ti,row in enumerate(texts):
            for seed in [1234,5678][:args.seeds]:
                t=time.perf_counter();request.infer(feeds(row,seed,names));y=np.array(request.get_output_tensor(0).data,copy=True)
                elapsed=time.perf_counter()-t;name=f"{row['id']}-s{seed}.wav"
                info=save_wav(out/name,y)
                one={'case':row['id'],'text':row['text'],'phoneme_count':len(row['phoneme_ids']),'seed':seed,
                     'wav':name,**info,'inference_pc_s':elapsed,'not_n100_rtf':True}
                rows.append(one);write(out/'audio.json',rows)
                print(json.dumps({'model':args.model,'variant':args.variant,'case':row['id'],'seed':seed,'seconds':info['seconds']}),flush=True)
        nodes=old.runtime_nodes(compiled);write(out/'runtime.json',nodes)
        kernels=old.int8_kernels(nodes);record['actual_int8_kernel_count']=len(kernels)
        record['actual_int8_kernels']=kernels
        if args.variant not in ['fp32'] and not kernels: record['status']='completed_without_int8_kernel'
        else: record['status']='audio_generated_quality_pending'
        record['artifacts']={x.name:sha(x) for x in out.iterdir() if x.suffix in ('.xml','.bin','.wav')}
        record['source_hash_unchanged']=sha(source)==item['fp32']['sha256']
    except BaseException:
        record['status']='failed';record['error']=traceback.format_exc();raise
    finally:
        record['finished_utc']=utc();write(out/'record.json',record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--run',default='screening-v1')
    p.add_argument('--seeds',type=int,default=2);p.add_argument('--native',action='store_true');p.add_argument('--cases')
    run_one(p.parse_args())
