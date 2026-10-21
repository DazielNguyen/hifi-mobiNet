"""Extracts architecture-relevant code with exact line numbers and file SHA-256.
Source = CURRENT working tree (BanhmiTTS HEAD 0a9bec2; only export_onnx.py is modified
in the tree) and the local Piper clone. Current code is NOT proof of the code that ran
during historical training; see Q03 section 11 / q03_q04_completion recovered_source.
Usage: python extract_code_excerpts.py <package_root>
"""
import hashlib
import json
import os
import subprocess
import sys

PKG = sys.argv[1]
ROOT = os.path.abspath(os.path.join(PKG, "..", ".."))  # FPT-Graduation-Project
B = os.path.join(ROOT, "BanhmiTTS")
P = os.path.join(ROOT, "piper", "src", "python", "piper_train", "vits")
SPANS = [
    ("BanhmiTTS", "banhmi_train/vits/modules/generator.py", [(18, 24), (25, 60), (61, 73), (74, 143), (145, 169)]),
    ("BanhmiTTS", "banhmi_train/vits/utils/resblocks.py", [(70, 106), (107, 175)]),
    ("BanhmiTTS", "banhmi_train/vits/utils/normalization.py", [(25, 90)]),
    ("BanhmiTTS", "banhmi_train/vits/modules/flow_block.py", [(14, 122)]),
    ("BanhmiTTS", "banhmi_train/vits/modules/synthesizer.py", [(88, 140), (216, 241), (250, 330)]),
    ("BanhmiTTS", "banhmi_train/vits/training.py", [(195, 220), (436, 480), (520, 620), (698, 730)]),
    ("BanhmiTTS", "banhmi_train/vits/utils/flows.py", [(1, 400)]),
    ("BanhmiTTS", "configs/banhmi_mrf.yaml", [(1, 200)]),
    ("BanhmiTTS", "configs/banhmi_mb_sequential_111.yaml", [(1, 200)]),
    ("BanhmiTTS", "configs/default.yaml", [(1, 200)]),
    ("piper", "models.py", [(212, 256), (299, 376), (630, 655)]),
    ("piper", "modules.py", [(220, 240), (323, 380), (412, 470)]),
]


def git(repo, *a):
    try:
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


meta = {"banhmitts_head": git(B, "rev-parse", "HEAD"), "banhmitts_status": git(B, "status", "--short", "--", "banhmi_train", "configs"),
        "piper_head": git(os.path.join(ROOT, "piper"), "rev-parse", "HEAD"), "files": []}
md = ["# Code excerpts (current working tree)", "",
      f"BanhmiTTS HEAD `{meta['banhmitts_head']}`; tracked changes under banhmi_train/configs: `{meta['banhmitts_status'] or 'none'}`.",
      f"Piper HEAD `{meta['piper_head']}`.",
      "Current source is not proof of historical training-time source (see Q03 §11).", ""]
for repo, rel, spans in SPANS:
    path = os.path.join(B if repo == "BanhmiTTS" else P, rel)
    if not os.path.exists(path):
        md.append(f"## {repo}:{rel} — MISSING")
        meta["files"].append({"repo": repo, "path": rel, "status": "MISSING"})
        continue
    data = open(path, "rb").read()
    lines = data.decode("utf-8").splitlines()
    h = hashlib.sha256(data).hexdigest()
    meta["files"].append({"repo": repo, "path": rel, "sha256": h, "n_lines": len(lines), "spans": spans})
    md.append(f"## {repo}:{rel}")
    md.append(f"SHA-256 `{h}`, {len(lines)} lines")
    for a, b in spans:
        b = min(b, len(lines))
        ext = "yaml" if rel.endswith(".yaml") else "python"
        md.append(f"\n### lines {a}–{b}\n```{ext}")
        md += [f"{i:4d}  {lines[i - 1]}" for i in range(a, b + 1)]
        md.append("```")
    md.append("")
out = os.path.join(PKG, "evidence", "D_architecture")
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "code_excerpts.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump(meta, open(os.path.join(out, "code_excerpts_index.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "files"}), len(meta["files"]), "files")
