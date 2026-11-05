"""Candidate common scope: ordinary Conv only in encoder, regular SDP and main flow.
All depthwise, MatMul, alignment, spline and decoder operations explicitly excluded.
Experiment only; does not change any original source/config.
"""
import run_experiment as e
import argparse
variants0=e.variants;scope0=e.scope
e.variants=lambda:variants0()+[{'id':'common97','mask':7}]
def scope(model,v):
    allowed,ignored,groups=scope0(model,v)
    if v['id']=='common97':
        types={n.get_friendly_name():n.get_type_name() for n in model.get_ops()}
        allowed=[name for name in allowed if types[name]=='Convolution'];assert len(allowed)==97
        ignored=sorted(n.get_friendly_name() for n in model.get_ops() if n.get_type_name() not in ('Constant','Parameter','Result') and n.get_friendly_name() not in allowed)
    return allowed,ignored,groups
e.scope=scope
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--native',action='store_true')
    a=p.parse_args();a.variant='common97';a.run='common97-native-v1' if a.native else 'common97-paired-v1';a.seeds=2;a.cases=None;e.run_one(a)
