"""Export one selected generator and verify FP32 ONNX, including nonzero noise.

The deployment graph keeps both random nodes. A separately labelled validation
graph replaces only those nodes with captured PyTorch noise inputs, allowing a
numerical comparison without disabling either stochastic component.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
import torch


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')


def metrics(a,b):
    if a.shape!=b.shape:return {'same_shape':False,'reference_shape':list(a.shape),'onnx_shape':list(b.shape),'passed':False}
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:return {'same_shape':True,'finite':False,'passed':False}
    err=a.astype(np.float64)-b.astype(np.float64)
    rmse=float(np.sqrt(np.mean(err**2)))
    norm=float(np.sqrt(np.mean(a.astype(np.float64)**2)))
    return {'same_shape':True,'finite':True,'shape':list(a.shape),'max_abs_error':float(np.max(np.abs(err))),
            'mean_abs_error':float(np.mean(np.abs(err))),'rmse':rmse,'reference_rms':norm,'nrmse':rmse/max(norm,1e-12),
            'passed':rmse/max(norm,1e-12)<=0.001 and float(np.max(np.abs(err)))<=0.005}


@contextmanager
def capture_noise():
    original_r,original_l=torch.randn,torch.randn_like
    captured=[]
    def randn(*a,**kw):
        value=original_r(*a,**kw);captured.append(('randn',value.detach().cpu().numpy().copy()));return value
    def randn_like(*a,**kw):
        value=original_l(*a,**kw);captured.append(('randn_like',value.detach().cpu().numpy().copy()));return value
    torch.randn,torch.randn_like=randn,randn_like
    try:yield captured
    finally:torch.randn,torch.randn_like=original_r,original_l


def session(path):
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
    opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('model',choices=['baseline','mrf','seq','piper'])
    args=parser.parse_args();root=Path(args.root);label=args.model
    config=json.loads((root/'RUN_CONFIG.json').read_text())
    assert config['thresholds_declared_before_runs']=={'nrmse_max':0.001,'max_abs_error_max':0.005,'same_shape_required':True,'finite_required':True}
    out=root/label;out.mkdir(exist_ok=False)
    report={'model':label,'started_utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'status':'running',
            'checkpoint':config['checkpoints'][label],'source_manifest_sha256':digest(root/'SOURCE_MANIFEST.json'),
            'runner_sha256':digest(__file__),'run_config_sha256':digest(root/'RUN_CONFIG.json'),
            'engineering_inputs_sha256':digest(root/'inputs/export_checks.json'),
            'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
                'onnx':onnx.__version__,'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider','threads':2,
                'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')},
            'not_quality_or_edge_benchmark':True,'thresholds':config['thresholds_declared_before_runs']}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
        family='piper' if label=='piper' else 'banhmi'
        sys.path.insert(0,str(root/'source'/family))
        if family=='piper':from piper_train.vits.models import SynthesizerTrn
        else:from banhmi_train.vits.modules.synthesizer import SynthesizerTrn
        ckpt=Path(report['checkpoint']['snapshot_checkpoint'])
        assert digest(ckpt)==report['checkpoint']['sha256']
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        hp=payload['hyper_parameters'];state=payload['state_dict']
        kwargs={k:hp[k] for k in inspect.signature(SynthesizerTrn.__init__).parameters if k in hp}
        kwargs.update(n_vocab=hp['num_symbols'],spec_channels=hp['filter_length']//2+1,
                      segment_size=hp['segment_size']//hp['hop_length'],n_speakers=1,gin_channels=0)
        if family!='piper':kwargs.update(use_f0=False,use_vocos=False)
        assert hp.get('num_speakers',1)<=1 and not hp.get('use_f0',False) and not hp.get('use_vocos',False)
        g=SynthesizerTrn(**kwargs).cpu().float().eval()
        weights={k.removeprefix('model_g.'):v for k,v in state.items() if k.startswith('model_g.')}
        loaded=g.load_state_dict(weights,strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        report.update(epoch=int(payload['epoch']),global_step=int(payload['global_step']),
                      generator_state_entries=len(weights),strict_load=True,generator_constructor=kwargs,
                      inference_parameters=sum(p.numel() for n,p in g.named_parameters() if not n.startswith('enc_q.')))
        save(out/'checkpoint_hparams.json',hp)
        del payload,state,weights
        selected=json.loads((root/'inputs/export_checks.json').read_text())['rows']
        def inputs(row,scales=(0.667,1.0,0.8)):
            text=torch.tensor([row['phoneme_ids']],dtype=torch.long)
            return (text,torch.tensor([text.shape[1]],dtype=torch.long),torch.tensor(scales,dtype=torch.float32))
        def forward(text,lengths,scales):
            kw={'noise_scale':scales[0],'length_scale':scales[1],'noise_scale_w':scales[2]}
            if family!='piper':kw['onnx_export']=True
            return g.infer(text,lengths,**kw)[0]
        dummy=inputs(selected[0])
        torch.manual_seed(1234)
        with torch.inference_mode():before=forward(*dummy).numpy()
        g.dec.remove_weight_norm()
        if hasattr(g.flow,'remove_weight_norm'):g.flow.remove_weight_norm()
        torch.manual_seed(1234)
        with torch.inference_mode():after=forward(*dummy).numpy()
        report['weight_norm_removal_check']=metrics(before,after)
        assert report['weight_norm_removal_check']['passed']
        g.forward=forward
        destination=out/(label+'_fp32.onnx')
        print('EXPORT',label,report['epoch'],flush=True)
        with torch.inference_mode():
            torch.onnx.export(g,dummy,str(destination),opset_version=18,dynamo=False,
                input_names=['input','input_lengths','scales'],output_names=['output'],
                dynamic_axes={'input':{0:'batch_size',1:'phonemes'},'input_lengths':{0:'batch_size'},'output':{0:'batch_size',2:'samples'}})
        graph=onnx.load(destination)
        helper.set_model_props(graph,{'q05_model':label,'checkpoint_sha256':report['checkpoint']['sha256'],
            'checkpoint_epoch_zero_based':str(report['epoch']),'source_manifest_sha256':report['source_manifest_sha256'],
            'output_contract':'float32 [batch, 1, samples]','q05_purpose':'FP32 engineering validation; quality and PTQ pending'})
        onnx.checker.check_model(graph,full_check=True)
        onnx.save(graph,destination)
        assert len(graph.graph.output[0].type.tensor_type.shape.dim)==3
        assert graph.graph.output[0].type.tensor_type.shape.dim[2].dim_param=='samples'
        assert not any('enc_q' in p.name for p in graph.graph.initializer)
        assert not any('QuantizeLinear' in n.op_type or 'QLinear' in n.op_type for n in graph.graph.node)
        assert not any(p.data_type==TensorProto.FLOAT16 for p in graph.graph.initializer)
        random_nodes=[n for n in graph.graph.node if n.op_type.startswith('Random')]
        assert len(random_nodes)==2 and all(n.op_type=='RandomNormalLike' for n in random_nodes)
        duration=next(n for n in random_nodes if '/dp/' in n.name)
        acoustic=next(n for n in random_nodes if n.name!=duration.name)
        report['graph']={'path':str(destination),'sha256':digest(destination),'bytes':destination.stat().st_size,
            'opset':18,'nodes':len(graph.graph.node),'initializers':len(graph.graph.initializer),
            'op_counts':dict(Counter(n.op_type for n in graph.graph.node)),
            'random_nodes':[{'name':n.name,'op':n.op_type,'output':list(n.output)} for n in random_nodes]}
        # Examine weights of historical FP32 graphs without assigning their provenance.
        if label in config['historical_fp32']:
            record=config['historical_fp32'][label];oldpath=Path(record['archived_path'])
            assert digest(oldpath)==record['sha256']
            oldgraph=onnx.load(oldpath);current_state=g.state_dict();matches=[];mismatches=[]
            for item in oldgraph.graph.initializer:
                if item.name not in current_state:continue
                a=numpy_helper.to_array(item);b=current_state[item.name].detach().numpy()
                if a.shape==b.shape and np.array_equal(a,b):matches.append(item.name)
                else:mismatches.append(item.name)
            report['historical_named_weight_comparison']={'source':record['source'],'sha256':record['sha256'],
                'exact_matches':len(matches),'mismatches':len(mismatches),'mismatch_names':mismatches,
                'limitation':'Named weights only; folded/renamed weights and graph semantics not exhaustively matched. No retrospective export-command or calibration provenance.'}
            del oldgraph,current_state
        # Validation graph differs ONLY by replacing the two random outputs with inputs.
        parity=onnx.ModelProto();parity.CopyFrom(graph)
        kept=[n for n in parity.graph.node if not n.op_type.startswith('Random')]
        del parity.graph.node[:];parity.graph.node.extend(kept)
        for node,channels,axis in [(duration,2,'phonemes'),(acoustic,kwargs['inter_channels'],'latent_frames')]:
            parity.graph.input.append(helper.make_tensor_value_info(node.output[0],TensorProto.FLOAT,['batch_size',channels,axis]))
        expected=onnx.ModelProto();expected.CopyFrom(graph)
        del expected.graph.node[:];expected.graph.node.extend([n for n in graph.graph.node if not n.op_type.startswith('Random')])
        stripped=onnx.ModelProto();stripped.CopyFrom(parity);del stripped.graph.input[-2:]
        assert stripped.SerializeToString()==expected.SerializeToString()
        paritypath=out/(label+'_validation_external_noise.onnx')
        onnx.checker.check_model(parity);onnx.save(parity,paritypath)
        report['validation_graph']={'path':str(paritypath),'sha256':digest(paritypath),'only_random_nodes_replaced':True,'deployment_artifact':False}
        parity_session=session(paritypath);report['parity_cases']=[]
        cases=[(r['id'],inputs(r),1234) for r in selected]
        cases += [('repeat_seed_'+selected[2]['id'],inputs(selected[2]),1235),
                  ('length_0.8_'+selected[2]['id'],inputs(selected[2],(0.667,0.8,0.8)),1234),
                  ('length_1.2_'+selected[2]['id'],inputs(selected[2],(0.667,1.2,0.8)),1234)]
        aa,bb=selected[0]['phoneme_ids'],selected[1]['phoneme_ids'];width=max(len(aa),len(bb))
        padded=torch.zeros((2,width),dtype=torch.long);padded[0,:len(aa)]=torch.tensor(aa);padded[1,:len(bb)]=torch.tensor(bb)
        cases.append(('padded_batch_2',(padded,torch.tensor([len(aa),len(bb)]),torch.tensor([0.667,1.0,0.8])),1234))
        for number,(identifier,args_,seed) in enumerate(cases):
            torch.manual_seed(seed)
            with torch.inference_mode(),capture_noise() as noise:
                reference=g(*args_).numpy()
            assert [kind for kind,_ in noise]==['randn','randn_like']
            assert all(np.std(v)>0 for _,v in noise)
            feeds={k:v.numpy() for k,v in zip(['input','input_lengths','scales'],args_)}
            feeds.update({duration.output[0]:noise[0][1],acoustic.output[0]:noise[1][1]})
            candidate=parity_session.run(None,feeds)[0]
            result=metrics(reference,candidate)
            arrays=out/f'parity_case_{number:02d}.npz'
            np.savez_compressed(arrays,input=args_[0].numpy(),input_lengths=args_[1].numpy(),scales=args_[2].numpy(),
                noise_duration=noise[0][1],noise_acoustic=noise[1][1],pytorch=reference,onnx=candidate)
            report['parity_cases'].append({'id':identifier,'seed':seed,'phoneme_lengths':args_[1].tolist(),
                'scales':args_[2].tolist(),'arrays':str(arrays),'arrays_sha256':digest(arrays),**result})
            save(out/'REPORT.json',report)
            print('PARITY',label,identifier,result,flush=True)
            assert result['passed'],f'Numerical parity failed: {identifier}'
        del parity_session
        native=session(destination);native_checks=[]
        for i,row in enumerate([selected[0],selected[3],selected[-1]]):
            args_=inputs(row);feeds={k:v.numpy() for k,v in zip(['input','input_lengths','scales'],args_)}
            wave=native.run(None,feeds)[0]
            valid=wave.ndim==3 and wave.shape[:2]==(1,1) and wave.shape[2]>4410 and bool(np.isfinite(wave).all()) and float(np.std(wave))>1e-6
            assert valid
            native_checks.append({'id':row['id'],'shape':list(wave.shape),'finite':True,'std':float(np.std(wave)),
                                  'sample_rate':hp['sample_rate'],'duration_seconds':wave.shape[2]/hp['sample_rate'],
                                  'wave_sha256':hashlib.sha256(wave.tobytes()).hexdigest()})
        del native
        # Runtime-specific RNG behavior, measured rather than assumed.
        feeds={k:v.numpy() for k,v in zip(['input','input_lengths','scales'],dummy)}
        ort.set_seed(1234);s1=session(destination);v1=s1.run(None,feeds)[0];v2=s1.run(None,feeds)[0];del s1
        ort.set_seed(1234);s2=session(destination);v3=s2.run(None,feeds)[0];del s2
        report['native_random_smoke']=native_checks
        report['ort_rng_check']={'seed':1234,'fresh_sessions_same_seed_equal':bool(np.array_equal(v1,v3)),
            'consecutive_calls_same_session_equal':bool(np.array_equal(v1,v2)),
            'scope':'This CPU runtime/provider and engineering input only; not a cross-framework noise equivalence claim.'}
        for r in json.loads((root/'SOURCE_MANIFEST.json').read_text()):
            assert digest(root/r['snapshot'])==r['sha256'],r['snapshot']
        assert digest(ckpt)==report['checkpoint']['sha256']
        report.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat())
        save(out/'REPORT.json',report)
        print('PASSED',label,report['graph']['sha256'],flush=True)
    except Exception:
        report.update(status='failed',error=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        save(out/'REPORT.json',report)
        raise


if __name__=='__main__':main()
