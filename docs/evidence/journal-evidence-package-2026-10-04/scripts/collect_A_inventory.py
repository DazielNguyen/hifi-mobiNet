"""Part A: environment, hardware and checkpoint metadata, re-read NOW (2026-10-04).
Everything here is "recomputed/re-inspected today", not metadata recorded at run time,
unless a field is read from inside a checkpoint (those fields were written by the
training process when the checkpoint was saved).
Run with <WSL_PROJECT_HOME>/env/bin/python. Writes evidence/A_inventory/*.json.
"""
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import subprocess
import sys

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def ver(pkgs):
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


env = {
    "inspected_at_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
    "note": "Current environment identity; does not prove versions used by historical runs.",
    "uname": platform.uname()._asdict(),
    "os_release": sh("grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release"),
    "cpu": sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread\\(s\\) per core|Core\\(s\\) per socket|Socket\\(s\\)|Flags' | sed 's/Flags:.*\\(avx512[a-z_]*\\|avx_vnni\\).*/Flags: (contains vnni\\/avx512 markers, see flags_vnni)/'"),
    "flags_vnni": sh("grep -o -w -E 'avx_vnni|avx512_vnni|avx512f|avx2' /proc/cpuinfo | sort | uniq -c"),
    "mem_total": sh("grep MemTotal /proc/meminfo"),
    "gpus": sh("nvidia-smi -L 2>/dev/null"),
    "main_env": {
        "python": sys.version,
        "executable": sys.executable,
        "packages": ver(["torch", "pytorch-lightning", "onnx", "onnxruntime", "numpy", "scipy", "librosa",
                         "openai-whisper", "utmosv2", "jiwer", "soundfile", "torchaudio", "transformers", "timm"]),
        "torch_default_num_threads_now": torch.get_num_threads(),
        "torch_default_interop_threads_now": torch.get_num_interop_threads(),
    },
    "piper_env": json.loads(sh(
        "<WSL_PROJECT_HOME>/piper_env/bin/python -c \"import sys,json,importlib.metadata as m,torch;"
        "v=lambda p: (m.version(p) if p in {d.metadata['Name'].lower() for d in m.distributions()} else None);"
        "print(json.dumps({'python':sys.version,'torch':torch.__version__,'pytorch-lightning':v('pytorch-lightning'),"
        "'torch_default_num_threads_now':torch.get_num_threads()}))\" 2>/dev/null") or "{}"),
}
json.dump(env, open(OUT / "environment_now.json", "w"), indent=1)

CKPTS = {
    "baseline (HiFi-GAN ResBlock2)": "<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt",
    "MRF (Parallel-IR)": "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt",
    "SEQ (Sequential-IR)": "<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt",
    "Piper (external published checkpoint)": "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt",
}
meta = {"note": "Fields under 'stored_in_checkpoint' were written by the training process; sha256/size/mtime are measured now."}
for name, p in CKPTS.items():
    st = os.stat(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cb = {k: {kk: (str(vv) if not isinstance(vv, (int, float, str, type(None), bool)) else vv)
              for kk, vv in v.items() if kk in ("monitor", "mode", "best_model_path", "best_model_score",
                                                  "last_model_path", "current_score", "save_top_k", "dirpath")}
          for k, v in (ck.get("callbacks") or {}).items() if isinstance(v, dict)}
    opt = [{"lr": g.get("lr"), "initial_lr": g.get("initial_lr"), "betas": g.get("betas"), "eps": g.get("eps"),
            "weight_decay": g.get("weight_decay"), "fused": g.get("fused")}
           for o in ck.get("optimizer_states", []) for g in o.get("param_groups", [])]
    sched = [{k: v for k, v in s.items() if k in ("gamma", "last_epoch", "base_lrs", "_last_lr")}
             for s in ck.get("lr_schedulers", [])]
    hp = {k: (str(v) if isinstance(v, pathlib.PurePath) else v) for k, v in dict(ck.get("hyper_parameters", {})).items()}
    meta[name] = {
        "path": p, "size_bytes": st.st_size, "sha256_now": sha(p),
        "stored_in_checkpoint": {
            "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "pytorch_lightning_version": ck.get("pytorch-lightning_version"),
            "callbacks": cb, "optimizer_param_groups": opt, "lr_schedulers": sched,
            "hyper_parameters": hp,
            "state_dict_top_modules": sorted({k.split(".")[0] + "." + k.split(".")[1] if k.startswith("model_g.") else k.split(".")[0]
                                              for k in ck["state_dict"]}),
        },
    }
    del ck
json.dump(meta, open(OUT / "checkpoint_metadata_now.json", "w"), indent=1, default=str)
print("A done:", {n: (m["sha256_now"][:12], m["stored_in_checkpoint"]["epoch"]) for n, m in meta.items() if n != "note"})
