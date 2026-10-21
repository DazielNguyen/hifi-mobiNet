"""Parameter counts per module straight from checkpoint state_dicts, counted
the same way for Piper and BanhmiTTS. Weight-norm pairs are folded
(weight_g dropped, weight_v counted once) -- i.e. the count after
remove_weight_norm(), which is what inference / ONNX export carries.
"""
import os
import pathlib
from collections import defaultdict

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

CKPTS = {
    "piper": "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt",
    "baseline": "<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt",
    "mrf": "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt",
    "seq": "<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt",
}
INFER = ["enc_p", "dp", "flow", "dec", "emb_g"]  # what ONNX export keeps

for name, path in CKPTS.items():
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    raw = defaultdict(int)
    folded = defaultdict(int)
    for k, v in sd.items():
        parts = k.split(".")
        top = parts[0]
        sub = parts[1] if top == "model_g" else ""
        key = f"{top}.{sub}" if sub else top
        raw[key] += v.numel()
        if k.endswith("weight_g"):
            continue
        folded[key] += v.numel()
    infer = sum(folded[f"model_g.{m}"] for m in INFER)
    total_g = sum(v for k, v in folded.items() if k.startswith("model_g."))
    disc = {k: v for k, v in folded.items() if not k.startswith("model_g.")}
    print(f"\n=== {name} ({os.path.getsize(path)/2**20:.1f} MiB ckpt) ===")
    for k in sorted(folded):
        print(f"  {k:22s} {folded[k]:>12,}")
    print(f"  {'generator total':22s} {total_g:>12,}")
    print(f"  {'inference (no enc_q)':22s} {infer:>12,}  -> FP32 {infer*4/2**20:.2f} MiB")
    print(f"  {'decoder only':22s} {folded['model_g.dec']:>12,}  -> FP32 {folded['model_g.dec']*4/2**20:.2f} MiB")
    print(f"  {'discriminators':22s} {sum(disc.values()):>12,}  {disc}")
