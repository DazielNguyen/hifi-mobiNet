import sys,json,os
from pathlib import Path
import numpy as np
import torch,onnxruntime as ort
root=Path.cwd()
sys.path.insert(0,str(root/'scripts/evaluation/onnx'))
import common as C
out=root/'training_output/piper-low-eval-launch-2026-11-02'
base=root/'training_output/piper-low-eval-2026-11-02'
graph=base/'graphs/piper-low-no-vits2-cpn_fp32_matched_noise.onnx'
ckpt=root/'training_output/piper-low-no-vits2-cpn-2026-10-30/checkpoints/best-epoch=1351-val_loss_mel=19.1691.ckpt'
module,info=C.load_module(root/'configs/training/piper-low-no-vits2-cpn.yaml',ckpt)
s=json.loads((root/'results/manifests/harvard720.json').read_text())[719]
x=torch.LongTensor(s['phoneme_ids']).unsqueeze(0);xl=torch.LongTensor([x.shape[1]])
refs={};noise_by_thread={}
for t in [2,12]:
 torch.set_num_threads(t);torch.manual_seed(C.SEED)
 with torch.no_grad(),C.capture_noise() as noise: refs[str(t)]=module(x,xl,list(C.SCALES)).numpy()
 noise_by_thread[str(t)]=noise
C.export_wrapper(module.model_g)
C.fold_flow_weight_norm(module.model_g)
for t in [2,12]:
 torch.set_num_threads(t);torch.manual_seed(C.SEED)
 with torch.no_grad(): refs['folded_'+str(t)]=module(x,xl,list(C.SCALES)).numpy()
record={'idx':719,'text':s['text'],'thresholds':C.PARITY,'references':{},'onnx':[]}
for name,w in refs.items():record['references'][name]=C.parity(refs['12'],w)
for t in [1,2,12]:
 for optimize in [True,False]:
  opt=ort.SessionOptions();opt.intra_op_num_threads=t;opt.inter_op_num_threads=1;opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
  if not optimize:opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  sess=ort.InferenceSession(str(graph),sess_options=opt,providers=['CPUExecutionProvider'])
  dn=next(i.name for i in sess.get_inputs() if i.name not in ('input','input_lengths','scales',C.NOISE_INPUT))
  noise=noise_by_thread['12']
  feeds={'input':x.numpy(),'input_lengths':xl.numpy(),'scales':np.array(C.SCALES,np.float32),dn:noise[0][1],C.NOISE_INPUT:C.acoustic_buffer(noise[1][1],719)}
  got=sess.run(None,feeds)[0]
  record['onnx'].append({'threads':t,'optimization':optimize,'vs_reference':{n:C.parity(v,got) for n,v in refs.items()}})
  print(t,optimize,{n:C.parity(v,got)['nrmse'] for n,v in refs.items()},flush=True)
(out/'parity_probe.json').write_text(json.dumps(record,indent=1)+'\n')
print('PyTorch references',record['references'])
