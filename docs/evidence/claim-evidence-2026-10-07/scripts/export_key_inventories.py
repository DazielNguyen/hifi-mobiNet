"""Export parameter/buffer NAME inventories for J-C001 (no checkpoint is unpickled).

    python export_key_inventories.py --baseline-ckpt PATH --piper-ckpt PATH \
        --hifi-repo PATH_TO_HIFI_MOBINET_CLONE --output key_inventories.json

Checkpoint names are read statically from the zip member data.pkl with
pickletools.genops (string opcodes only; no object is reconstructed). Code-side
names come from state_dict() of randomly initialised models built from the
hifi-mobiNet clone (current BanhmiTTS-relocated code and vendored EdgeTTS code).
Only names are compared: tensor shapes and values are NOT compared.
Requires the clone's training environment (torch, pytorch_lightning, built MAS).
"""
import argparse
import hashlib
import json
import os
import pickletools
import re
import sys
import zipfile

NAME = re.compile(r"\.(weight|bias|weight_g|weight_v|gamma|beta|emb_rel_k|emb_rel_v|logs|m|alpha|_n_channels|hann_window)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def checkpoint_names(path):
    z = zipfile.ZipFile(path)
    data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
    return sorted({a for op, a, _ in pickletools.genops(data)
                   if isinstance(a, str) and a.startswith("model_") and NAME.search(a)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--piper-ckpt", required=True)
    p.add_argument("--hifi-repo", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    os.environ["HIFIMOBINET_HOME"] = a.hifi_repo
    sys.path.insert(0, os.path.join(a.hifi_repo, "src"))
    import warnings
    warnings.filterwarnings("ignore")
    from hifimobinet.training.config import load_config
    from hifimobinet.training.registry import get_training_model
    from hifimobinet.training.vendor import vendor_module

    cfg = load_config(os.path.join(a.hifi_repo, "configs/training/baseline-resblock2-vits2.yaml"))
    banhmi = sorted(get_training_model(cfg.model_id).builder()(num_symbols=256, **cfg.flat_hparams()).state_dict())
    edgetts = vendor_module("edgetts_vits", "lightning").VitsModel
    edg_v2 = sorted(edgetts(num_symbols=256, num_speakers=1, dataset=None, use_vits2=True).state_dict())
    edg_a = sorted(edgetts(num_symbols=256, num_speakers=1, dataset=None).state_dict())
    inv = {
        "baseline_checkpoint": {"sha256": sha256(a.baseline_ckpt), "names": checkpoint_names(a.baseline_ckpt)},
        "piper_checkpoint": {"sha256": sha256(a.piper_ckpt), "names": checkpoint_names(a.piper_ckpt)},
        "code_banhmitts_baseline_recipe": {"source": "src/hifimobinet/training/baseline_module.py (relocated BanhmiTTS model code)", "names": banhmi},
        "code_edgetts_use_vits2": {"source": "vendor/edgetts/vits/lightning.py use_vits2=True", "names": edg_v2},
        "code_edgetts_config_a": {"source": "vendor/edgetts/vits/lightning.py all flags off (Piper base)", "names": edg_a},
    }
    pairs = {"baseline_checkpoint_vs_code_banhmitts_baseline_recipe": ("baseline_checkpoint", "code_banhmitts_baseline_recipe"),
             "baseline_checkpoint_vs_code_edgetts_use_vits2": ("baseline_checkpoint", "code_edgetts_use_vits2"),
             "piper_checkpoint_vs_code_edgetts_config_a": ("piper_checkpoint", "code_edgetts_config_a")}
    comparisons = {}
    for label, (l, r) in pairs.items():
        L, R = set(inv[l]["names"]), set(inv[r]["names"])
        comparisons[label] = {"left_count": len(L), "right_count": len(R), "common": len(L & R),
                              "only_left": sorted(L - R), "only_right": sorted(R - L)}
    out = {"comparison_basis": "parameter and persistent-buffer NAMES only; shapes and values not compared",
           "name_filter_regex": NAME.pattern, "inventories": inv, "comparisons": comparisons}
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {x: v[x] for x in ("left_count", "right_count", "common")} for k, v in comparisons.items()}, indent=1))


if __name__ == "__main__":
    main()
