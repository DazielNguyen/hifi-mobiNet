"""One-intervention control: bypass just cumulative-duration FQ in historical graph copy.
Weights, other FQs, calibration constants and native RNG are unchanged.
"""
from legacy_pair_probe import *

if __name__=='__main__':
    os.nice(10)
    for model_id in MODELS:
        for mode in ['paired_bypass_cumsum','native_bypass_cumsum','paired_original_trace']:
            out=BASE/'cumsum-causal-v1'/model_id/mode;out.mkdir(parents=True,exist_ok=False)
            core=ov.Core();src=LEGACY/model_id/'int8.xml';model=core.read_model(str(src))
            before=inventory(model);fq=next(n for n in model.get_ops() if n.get_friendly_name()=='/CumSum/fq_output_0')
            maximum=float(np.max(fq.input_value(2).get_node().get_data()))
            intervention=[]
            if 'bypass' in mode:
                fq.output(0).replace(fq.input_value(0));intervention=['/CumSum/fq_output_0']
            names=None
            if mode.startswith('paired'):names,_=gaussian_inputs(model)
            after=inventory(model)
            b={r['name']:{k:v for k,v in r.items() if k!='consumers'} for r in before}
            a={r['name']:{k:v for k,v in r.items() if k!='consumers'} for r in after}
            assert set(b)-set(a)==set(intervention)
            assert all(a[k]==b[k] for k in a)
            ov.save_model(model,str(out/'model.xml'),compress_to_fp16=False)
            trace={}
            # Additional outputs are debug-only; do not use this graph for timing.
            byname={n.get_friendly_name():n for n in model.get_ops()}
            for name in ['/CumSum','/CumSum/fq_output_0','/Sub','/Sub/fq_output_0']:
                if name in byname:
                    trace[name]=len(model.outputs);model.add_outputs([byname[name].output(0)])
            compiled=core.compile_model(model,'CPU',CONFIG);request=compiled.create_infer_request();rows=[];traces=[]
            for row in read(BASE/'texts.json'):
                for seed in [1234,5678]:
                    request.infer(feeds(row,seed,names));y=np.array(request.get_output_tensor(0).data,copy=True)
                    name=f"{row['id']}-s{seed}.wav";info=save_wav(out/name,y)
                    rows.append({'case':row['id'],'text':row['text'],'seed':seed,'wav':name,**info});write(out/'audio.json',rows)
                    record={'case':row['id'],'seed':seed,'seconds':info['seconds'],'fq_input_high':maximum,'fq_high_seconds':maximum*256/22050}
                    for name,index in trace.items():
                        v=np.array(request.get_output_tensor(index).data,copy=True)
                        d={'shape':list(v.shape),'min':float(v.min()),'max':float(v.max())}
                        if 'CumSum' in name:d['values']=v.reshape(-1).tolist()
                        else:
                            # generate_path mask shape is [batch,phonemes,frames].
                            coverage=np.sum(v,axis=-2).reshape(-1)
                            d['frame_coverage']=coverage.tolist();d['zero_frames']=int(np.sum(coverage==0))
                        record[name]=d
                    traces.append(record);write(out/'alignment-trace.json',traces)
            nodes=old.runtime_nodes(compiled);write(out/'runtime.json',nodes)
            write(out/'record.json',{'status':'audio_generated_quality_pending','intervention':intervention,'other_fq_values_inputs_unchanged':True,
                'source_xml_sha256':sha(src),'source_bin_sha256':sha(src.with_suffix('.bin')),'original_fq_count':len(before),'remaining_fq_count':len(after),
                'actual_int8_kernel_count':len(old.int8_kernels(nodes)),'rng':'explicit paired inputs' if names else 'native (seed label is repeat label only)',
                'config':CONFIG,'script_sha256':sha(__file__),'not_n100_benchmark':True})
            print(model_id,mode,'DONE',flush=True)
