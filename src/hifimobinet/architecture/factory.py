"""Construct selected Banhmi architectures for inspection, without loading weights."""
import json
from pathlib import Path
from hifimobinet.registry import model_record,repository_root,within


def build_synthesizer(model_id: str, root: Path | None = None):
    if model_id not in {'baseline-resblock2','parallel-ir','sequential-ir'}:
        raise ValueError('This factory only describes the three Banhmi architectures; Piper uses its own reference source')
    root=root or repository_root()
    record=model_record(model_id,root)
    h=json.loads(within(root,record['config']).read_text())['parameters']
    if h.get('use_f0',False) or h.get('use_vocos',False):
        raise ValueError('F0/Vocos are outside the selected research scope')
    from .vits.modules.synthesizer import SynthesizerTrn
    required=['inter_channels','hidden_channels','filter_channels','n_heads','n_layers','kernel_size','p_dropout',
              'resblock','resblock_kernel_sizes','resblock_dilation_sizes','upsample_rates','upsample_initial_channel','upsample_kernel_sizes']
    optional=['posterior_encoder_kernel_size','posterior_encoder_dilation_rate','posterior_encoder_layers',
              'flow_kernel_size','flow_dilation_rate','flow_n_flows','mb_expansion']
    kwargs={k:h[k] for k in required}
    kwargs.update({k:h[k] for k in optional if k in h})
    return SynthesizerTrn(n_vocab=h['num_symbols'],spec_channels=h['filter_length']//2+1,
                         segment_size=h['segment_size']//h['hop_length'],n_speakers=1,
                         gin_channels=0,use_f0=False,use_vocos=False,**kwargs)
