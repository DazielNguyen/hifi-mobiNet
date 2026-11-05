import os,sys,json
from pathlib import Path
os.environ['NNCF_TELEMETRY_DISABLED']='1'
os.environ['OPENVINO_TELEMETRY_DISABLED']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['ONEDNN_MAX_CPU_ISA']='AVX2'
os.environ['DNNL_MAX_CPU_ISA']='AVX2'
sys.modules['openvino_telemetry']=None
root=Path.cwd()
sys.path[:0]=[str(root/'src'),str(root/'scripts/evaluation/edge')]
import openvino_suite as suite
original_env=suite.environment
suite.environment=lambda: {**original_env(require_target=False),'validation_host_only':True,'not_n100_benchmark':True}
bundle=root/'results/edge-n100-openvino/2026-11-01/.local/originals/final'
work=root/'.local/common97-integration-20261105/work'
suite.prepare(bundle,work,'piper')
print('INTEGRATION_ALL_THREE_PREPARED_ON_PC_NOT_N100',flush=True)
