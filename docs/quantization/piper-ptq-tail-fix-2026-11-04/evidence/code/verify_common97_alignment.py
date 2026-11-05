"""Observe alignment of candidate copies; no quantization and no source mutation."""
from legacy_pair_probe import *
rows=[]
for model_id in MODELS:
    core=ov.Core();model=core.read_model(str(BASE/'common97-paired-v1'/model_id/'common97/model.xml'))
    byname={n.get_friendly_name():n for n in model.get_ops()}
    assert '/CumSum/fq_output_0' not in byname
    model.add_outputs([byname['/CumSum'].output(0),byname['/Sub'].output(0)])
    compiled=core.compile_model(model,'CPU',CONFIG);request=compiled.create_infer_request()
    for row in read(BASE/'texts.json'):
        for seed in [1234,5678]:
            request.infer(feeds(row,seed,{'duration_input':'/dp/RandomNormalLike_output_0','acoustic_input':'noise_acoustic'}))
            c=np.array(request.get_output_tensor(1).data);alignment=np.array(request.get_output_tensor(2).data)
            coverage=alignment.sum(axis=-2).reshape(-1)
            r={'model':model_id,'case':row['id'],'seed':seed,'cumulative_frames':float(c.max()),'audio_frames':coverage.size,
               'zero_alignment_frames':int(np.sum(coverage==0)),'min_alignment_column_sum':float(coverage.min()),'max_alignment_column_sum':float(coverage.max())}
            rows.append(r)
            write(BASE/'common97-alignment-verification.json',rows)
            assert r['zero_alignment_frames']==0 and np.all(coverage==1),r
print('ALL_36_CANDIDATE_ALIGNMENTS_VALID',flush=True)
