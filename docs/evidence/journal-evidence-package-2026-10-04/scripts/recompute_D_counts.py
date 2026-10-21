"""RECOMPUTED parameter and decoder FLOP counts from the four checkpoints.

Parameter boundary: generator state_dict (model_g.*), weight-norm pairs folded
(weight_g dropped, weight_v counted once) = the tensors an inference export carries.
Buffers are not in the counts below unless they are state_dict entries; discriminators listed separately.

FLOP convention (same as the historical /tmp/count_flops_seq.py, re-implemented):
  MACs(Conv1d) = out_elements * in_channels/groups * kernel
  MACs(ConvTranspose1d) = in_elements * out_channels/groups * kernel
  FLOPs = 2 * MACs. Elementwise ops (SnakeBeta, LeakyReLU, tanh, adds, bias) are NOT counted.
  Decoder only (model_g.dec), batch 1, input latent [1, 192, T].
  Reported for T=86 frames (historical convention: 86*256 = 22016 samples = 0.99846 s)
  and per exact second of audio (MACs per frame * 22050/256).
Usage:
  main env : recompute_D_counts.py banhmi <out.json>
  piper env: recompute_D_counts.py piper  <out.json>
"""
import json
import pathlib
import sys

import torch
from torch import nn

torch.serialization.add_safe_globals([pathlib.PosixPath])
MODE, OUT = sys.argv[1], sys.argv[2]
HOP, SR = 256, 22050


def fold_counts(sd):
    per = {}
    for k, v in sd.items():
        if k.endswith("weight_g"):
            continue
        top = k.split(".")[0]
        key = f"{top}.{k.split('.')[1]}" if top == "model_g" else top
        per[key] = per.get(key, 0) + v.numel()
    return per


def dec_macs(dec, T):
    tot = {"conv1d": 0, "convtranspose1d": 0}
    hs = []

    def c1(m, i, o):
        tot["conv1d"] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def ct(m, i, o):
        tot["convtranspose1d"] += i[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0]

    for m in dec.modules():
        if isinstance(m, nn.ConvTranspose1d):
            hs.append(m.register_forward_hook(ct))
        elif isinstance(m, nn.Conv1d):
            hs.append(m.register_forward_hook(c1))
    with torch.no_grad():
        y = dec(torch.randn(1, 192, T))
    for h in hs:
        h.remove()
    return tot["conv1d"] + tot["convtranspose1d"], tot, tuple(y.shape)


if MODE == "banhmi":
    sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
    from banhmi_train.vits.training import VitsModel
    CK = {
        "baseline (HiFi-GAN ResBlock2)": "<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt",
        "MRF (Parallel-IR)": "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt",
        "SEQ (Sequential-IR)": "<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt",
    }
    load = lambda p: VitsModel.load_from_checkpoint(p, map_location="cpu", dataset=None, use_f0=False)
else:
    sys.path.insert(0, "<WORKSPACE>/piper/src/python")
    from piper_train.vits.lightning import VitsModel
    CK = {"Piper (external published checkpoint)": "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt"}
    load = lambda p: VitsModel.load_from_checkpoint(p, map_location="cpu", dataset=None)

res = {"mode": MODE, "torch": torch.__version__, "convention": __doc__}
for name, p in CK.items():
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    per = fold_counts(sd)
    infer = sum(per.get(f"model_g.{m}", 0) for m in ("enc_p", "dp", "flow", "dec", "emb_g"))
    model = load(p)
    model.eval()
    dec = model.model_g.dec
    with torch.no_grad():
        dec.remove_weight_norm()
    live = sum(x.numel() for x in dec.parameters())
    m86, parts, yshape = dec_macs(dec, 86)
    m1, _, _ = dec_macs(dec, 1000)
    per_frame = m1 / 1000
    res[name] = {
        "checkpoint": p,
        "params_folded_by_module": per,
        "inference_params_enc_p_dp_flow_dec": infer,
        "decoder_params_folded_state_dict": per.get("model_g.dec"),
        "decoder_params_live_after_remove_weight_norm": live,
        "posterior_encoder_training_only": per.get("model_g.enc_q"),
        "discriminators_training_only": {k: v for k, v in per.items() if not k.startswith("model_g.")},
        "decoder_macs_T86": m86, "decoder_macs_T86_parts": parts, "decoder_output_shape_T86": yshape,
        "decoder_gflops_T86": 2 * m86 / 1e9,
        "decoder_macs_per_frame_T1000": per_frame,
        "decoder_gflops_per_exact_second": 2 * per_frame * SR / HOP / 1e9,
    }
    del model, sd
json.dump(res, open(OUT, "w"), indent=1, default=str)
for k, v in res.items():
    if isinstance(v, dict) and "decoder_params_folded_state_dict" in v:
        print(f"{k}: dec_params={v['decoder_params_folded_state_dict']:,} (live {v['decoder_params_live_after_remove_weight_norm']:,}) "
              f"infer={v['inference_params_enc_p_dp_flow_dec']:,} GFLOPs@T86={v['decoder_gflops_T86']:.4f} "
              f"GFLOPs/exact-s={v['decoder_gflops_per_exact_second']:.4f} flow={v['params_folded_by_module'].get('model_g.flow'):,}")
