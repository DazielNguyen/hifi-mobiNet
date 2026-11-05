"""Control: preserve historical quantization, replace ONLY its two Gaussian RNG outputs.
No original file is written. This separates RNG from quantization/calibration changes.
"""
from run_experiment import *
from openvino import opset13 as ops

def gaussian_inputs(model):
    nodes={n.get_friendly_name():n for n in model.get_ops()}
    params=[]
    names={'duration_input':'/dp/RandomNormalLike_output_0','acoustic_input':'noise_acoustic'}
    replacements=[]
    for name,key,shape in [('/dp/RandomNormalLike','duration_input',[1,2,-1]),('/RandomNormalLike','acoustic_input',[1,192,10000])]:
        n=nodes[name]
        assert n.get_type_name()=='Add'
        # Find the pre-existing ShapeOf that feeds the Box-Muller RNG subgraph.
        queue=[n];seen=set();shapes=[]
        while queue:
            a=queue.pop()
            if a.get_instance_id() in seen: continue
            seen.add(a.get_instance_id())
            if a.get_type_name()=='ShapeOf': shapes.append(a);continue
            queue.extend(x.get_source_output().get_node() for x in a.inputs())
        assert len(shapes)==1,[(x.get_friendly_name(),x.get_type_name()) for x in shapes]
        p=ops.parameter(shape,np.float32,name=names[key]);p.output(0).set_names({names[key]});params.append(p)
        replacement=p
        if key=='acoustic_input':
            # Frames are taken from original like tensor; never from replaced Gaussian.
            frame=ops.gather(shapes[0],ops.constant([2],np.int64),ops.constant(0,np.int64))
            replacement=ops.slice(p,ops.constant([0],np.int64),frame,ops.constant([1],np.int64),ops.constant([2],np.int64))
        n.output(0).replace(replacement.output(0))
        replacements.append({'original_gaussian':name,'shape_source':shapes[0].input_value(0).get_node().get_friendly_name(),'new_parameter':names[key]})
    model.add_parameters(params);model.validate_nodes_and_infer_types()
    return names,replacements

def inventory(model):
    rows=[]
    for n in model.get_ordered_ops():
        if n.get_type_name()!='FakeQuantize':continue
        rows.append({'name':n.get_friendly_name(),'input':n.input_value(0).get_node().get_friendly_name(),
                     'consumers':[x.get_node().get_friendly_name() for x in n.output(0).get_target_inputs()],
                     'levels':n.get_levels(),
                     'ranges':[np.array(n.input_value(i).get_node().get_data()).tolist() if n.input_value(i).get_node().get_type_name()=='Constant' else n.input_value(i).get_node().get_friendly_name() for i in range(1,5)]})
    return rows

if __name__=='__main__':
    os.nice(10)
    for model_id in MODELS:
        for precision in ['fp32','int8']:
            out=BASE/'legacy-paired-v3'/model_id/precision;out.mkdir(parents=True,exist_ok=False)
            src=LEGACY/model_id/(precision+'.xml');core=ov.Core();model=core.read_model(str(src))
            source_hashes={str(p):sha(p) for p in [src,src.with_suffix('.bin')]}
            before=inventory(model);names,replacements=gaussian_inputs(model);after=inventory(model)
            # Consumer order can change; dead ShapeOf branches of removed RNG disappear.
            # Require every FQ name, data input, levels and four ranges to stay exact.
            canonical=lambda rows: sorted([{k:v for k,v in r.items() if k!='consumers'} for r in rows],key=lambda r:r['name'])
            assert canonical(before)==canonical(after),'Gaussian replacement changed FQ values/inputs'
            write(out/'fq-inventory.json',after)
            ov.save_model(model,str(out/'model.xml'),compress_to_fp16=False)
            compiled=core.compile_model(model,'CPU',CONFIG);request=compiled.create_infer_request()
            parity=[];reference=None
            if precision=='fp32':
                reference=core.compile_model(str(BASE/'full-v1'/model_id/'fp32/model.xml'),'CPU',CONFIG) if (BASE/'full-v1'/model_id/'fp32/model.xml').exists() else None
            rows=[]
            for row in read(BASE/'texts.json'):
                for seed in [1234,5678]:
                    feed=feeds(row,seed,names);request.infer(feed);y=np.array(request.get_output_tensor(0).data,copy=True)
                    name=f"{row['id']}-s{seed}.wav";info=save_wav(out/name,y)
                    rows.append({'case':row['id'],'text':row['text'],'seed':seed,'wav':name,**info});write(out/'audio.json',rows)
                    if reference:
                        # ONNX helper and this IR substitution must agree numerically.
                        r=reference.create_infer_request();r.infer(feed);z=np.array(r.get_output_tensor(0).data,copy=True)
                        assert y.shape==z.shape
                        parity.append({'case':row['id'],'seed':seed,'max_abs':float(np.max(np.abs(y-z))),'rmse':float(np.sqrt(np.mean((y.astype(float)-z)**2)))})
            nodes=old.runtime_nodes(compiled);write(out/'runtime.json',nodes)
            record={'status':'audio_generated_quality_pending','historical_precision':precision,'source_hashes':source_hashes,
                    'only_graph_change':'Replace two Gaussian RNG outputs by explicit duration/acoustic inputs; preserve FQ nodes/ranges',
                    'replacements':replacements,'fake_quantize_count':len(before),'actual_int8_kernel_count':len(old.int8_kernels(nodes)),
                    'paired_noise_parity_against_onnx_helper':parity,'config':CONFIG,'not_n100_benchmark':True,'script_sha256':sha(__file__),
                    'source_hashes_unchanged':all(sha(p)==h for p,h in source_hashes.items())}
            write(out/'record.json',record);print(model_id,precision,'DONE',flush=True)
