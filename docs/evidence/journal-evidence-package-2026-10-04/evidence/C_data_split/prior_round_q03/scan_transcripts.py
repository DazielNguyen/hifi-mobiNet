"""Timeline of (a) edits to the training/split code and (b) training launches /
resumes, taken from Claude Code session transcripts (timestamped JSONL).
Read-only; prints one line per relevant event, oldest first."""
import glob
import json
import os
import re
import sys

ROOT = os.path.expanduser(
    r"~/.claude/projects/<CLAUDE_PROJECT_SLUG>")
CODE_FILES = ("training.py", "dataset.py", "train.py", "synthesizer.py", "generator.py",
              "flow_block.py", "duration_predictor.py", "resblocks.py", "normalization.py")
SPLIT_RE = re.compile(r"random_split|canonical_split|Generator\(\)|manual_seed|_split\b", re.I)
LAUNCH_RE = re.compile(r"(nohup|python -m banhmi_train\.train|launch_\w+\.sh|resume_\w+\.sh)", re.I)
RUN_RE = re.compile(r"(baseline_v2_noclip|mrf_111_run_500ep|seq111_final)")

events = []
for path in glob.glob(os.path.join(ROOT, "*.jsonl")):
    sess = os.path.basename(path)[:8]
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            ts = e.get("timestamp")
            msg = e.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if not ts or not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict) or c.get("type") != "tool_use":
                    continue
                name, inp = c.get("name"), c.get("input") or {}
                if name in ("Edit", "Write", "MultiEdit"):
                    fp = inp.get("file_path", "")
                    if "BanhmiTTS" not in fp or not fp.endswith(".py"):
                        continue
                    body = json.dumps(inp, ensure_ascii=False)
                    tag = "SPLIT-EDIT" if SPLIT_RE.search(body) else "code-edit"
                    snippet = (inp.get("new_string") or inp.get("content") or "")[:160].replace("\n", " ")
                    events.append((ts, sess, tag, f"{name} {os.path.basename(fp)} :: {snippet}"))
                elif name in ("Bash", "PowerShell"):
                    cmd = inp.get("command", "")
                    if RUN_RE.search(cmd) and LAUNCH_RE.search(cmd) and "eval" not in cmd.lower():
                        events.append((ts, sess, "LAUNCH", cmd[:260].replace("\n", " | ")))
                    elif re.search(r"cp .*banhmi_train|rsync .*BanhmiTTS|cp -r .*BanhmiTTS|resume_baseline|launch_baseline", cmd):
                        events.append((ts, sess, "SYNC", cmd[:200].replace("\n", " | ")))

events.sort()
only = sys.argv[1] if len(sys.argv) > 1 else None
for ts, sess, tag, txt in events:
    if only and tag != only and not (only == "key" and tag in ("SPLIT-EDIT", "LAUNCH")):
        continue
    print(f"{ts[:19]}Z [{sess}] {tag:10s} {txt}")
