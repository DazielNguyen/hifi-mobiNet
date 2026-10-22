"""Does a system's ASR error concentrate on words whose phonemes differ between two espeak voices?

Exploratory check of the frontend-mismatch hypothesis (results/frontend-check/README.md).
Harvard-720 feeds en-us espeak phonemes to every system; the released config of the external
Piper checkpoint names espeak voice "en". If a system learned the other voice's phonemes, its
errors should rise with the per-sentence phoneme distance between the two voices and fall on
the words whose phonemes differ, more than for a control system trained on the fed voice.

Nothing is synthesized or scored: the script re-phonemizes the Harvard text with both voices
(``banhmi_phonemize``, the frontend that built the manifest) and reads stored per-sentence
results. Word errors are taken from a JiWER alignment with the scorer's WER transform.

    python scripts/evaluation/harvard/dialect_check.py \
        --candidate piper=results/historical/harvard/harvard_piper_results.json \
        --control piper_no_vits2_cpn=results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
        --output results/frontend-check/dialect_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

N_BOOT, SEED = 5000, 20261023
PUNCT = set(".,;:!?\"'")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edit_distance(a, b) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def phoneme_words(ph):
    out, cur = [], []
    for p in ph:
        if p == " ":
            if cur:
                out.append("".join(cur))
                cur = []
        elif p not in PUNCT:
            cur.append(p)
    if cur:
        out.append("".join(cur))
    return [w.replace("ˈ", "").replace("ˌ", "") for w in out]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--candidate", required=True, metavar="LABEL=PATH")
    p.add_argument("--control", required=True, metavar="LABEL=PATH")
    p.add_argument("--fed-voice", default="en-us")
    p.add_argument("--other-voice", default="en")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")

    import jiwer
    import numpy as np
    import scipy
    from banhmi_phonemize import phonemize_espeak
    from scipy.stats import spearmanr

    T = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                       jiwer.ReduceToListOfListOfWords()])
    labels, paths = {}, {}
    for role, spec in (("candidate", a.candidate), ("control", a.control)):
        lab, _, path = spec.partition("=")
        labels[role], paths[role] = lab, Path(path)
    rows = {r: json.loads(paths[r].read_text(encoding="utf-8")) for r in paths}
    man = json.loads(a.manifest.read_text(encoding="utf-8"))
    for r in rows.values():
        if [x["text"] for x in r] != [s["text"] for s in man]:
            raise SystemExit("results are not in manifest order")

    phon = lambda text, voice: [x for sent in phonemize_espeak(text, voice) for x in sent]  # noqa: E731
    fed_equals_manifest, sentences, words = 0, [], []
    for i, s in enumerate(man):
        fed, other = phon(s["text"], a.fed_voice), phon(s["text"], a.other_voice)
        fed_equals_manifest += fed == list(s["phonemes"])
        strip = lambda ph: [x for x in ph if x not in "ˈˌ"]  # noqa: E731
        dist = edit_distance(strip(fed), strip(other)) / max(len(strip(fed)), 1)
        ref = T(s["text"])[0]
        flags = {}
        for role, r in rows.items():
            out = jiwer.process_words(s["text"], r[i]["asr_text"], reference_transform=T, hypothesis_transform=T)
            f = [0] * len(ref)
            for al in out.alignments[0]:
                if al.type in ("substitute", "delete"):
                    for j in range(al.ref_start_idx, al.ref_end_idx):
                        f[j] = 1
            flags[role] = f
        sentences.append({"idx": i, "phoneme_distance": dist, "wer_candidate": rows["candidate"][i]["wer"],
                          "wer_control": rows["control"][i]["wer"]})
        wf, wo = phoneme_words(fed), phoneme_words(other)
        if len(wf) == len(wo) == len(ref):
            for j, w in enumerate(ref):
                words.append({"sentence": i, "word": w, "fed": wf[j], "other": wo[j], "differs": wf[j] != wo[j],
                              "error_candidate": flags["candidate"][j], "error_control": flags["control"][j]})

    d = np.array([x["phoneme_distance"] for x in sentences])
    wc = np.array([x["wer_candidate"] for x in sentences])
    wk = np.array([x["wer_control"] for x in sentences])
    sp = {name: dict(zip(("rho", "p"), map(float, spearmanr(d, y)))) for name, y in
          (("candidate_wer", wc), ("control_wer", wk), ("candidate_minus_control_wer", wc - wk))}
    cut = np.quantile(d, [1 / 3, 2 / 3])
    terc = np.digitize(d, cut)
    terciles = [{"tercile": g + 1, "n": int((terc == g).sum()), "mean_distance": float(d[terc == g].mean()),
                 "candidate_wer": float(wc[terc == g].mean()), "control_wer": float(wk[terc == g].mean())} for g in range(3)]

    def rates(ws):
        arr = np.array([[w["differs"], w["error_candidate"], w["error_control"]] for w in ws], float)
        dif, same = arr[arr[:, 0] == 1], arr[arr[:, 0] == 0]
        return {"words_differing": int(len(dif)), "words_same": int(len(same)),
                "candidate_error_rate_differing": float(dif[:, 1].mean()), "candidate_error_rate_same": float(same[:, 1].mean()),
                "control_error_rate_differing": float(dif[:, 2].mean()), "control_error_rate_same": float(same[:, 2].mean()),
                "candidate_errors_differing": int(dif[:, 1].sum()), "candidate_errors_same": int(same[:, 1].sum()),
                "control_errors_differing": int(dif[:, 2].sum()), "control_errors_same": int(same[:, 2].sum())}

    base = rates(words)
    did = lambda r: (r["candidate_error_rate_differing"] - r["candidate_error_rate_same"]) - (  # noqa: E731
        r["control_error_rate_differing"] - r["control_error_rate_same"])
    by = {}
    for w in words:
        by.setdefault(w["sentence"], []).append(w)
    keys = sorted(by)
    rng = np.random.default_rng(SEED)
    boot = [did(rates([w for k in rng.choice(keys, len(keys)) for w in by[k]])) for _ in range(N_BOOT)]
    excess_diff = base["candidate_errors_differing"] - base["control_errors_differing"]
    excess_same = base["candidate_errors_same"] - base["control_errors_same"]
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "classification": "exploratory_analysis_of_stored_results",
        "question": "frontend mismatch: do the candidate's ASR errors concentrate on words whose phonemes differ between the fed and the other espeak voice?",
        "inputs": {"manifest": {"file": str(a.manifest), "sha256": sha256(a.manifest)},
                   **{labels[r]: {"role": r, "file": str(paths[r]), "sha256": sha256(paths[r])} for r in paths}},
        "voices": {"fed": a.fed_voice, "other": a.other_voice},
        "fed_voice_rephonemization_equals_manifest": fed_equals_manifest,
        "environment": {"python": platform.python_version(), "jiwer": jiwer.__version__ if hasattr(jiwer, "__version__") else None,
                        "numpy": np.__version__, "scipy": scipy.__version__},
        "method": {"phoneme_distance": "edit distance between the two voices' phoneme sequences without stress marks, divided by the fed length",
                   "word_errors": "substituted or deleted reference words in the JiWER alignment (scorer transform)",
                   "word_level_sentences": "only sentences whose phoneme words align one-to-one with the reference words",
                   "difference_in_differences": "(candidate rate on differing - same words) - (control rate on differing - same words)",
                   "bootstrap": f"{N_BOOT} resamples of sentences, default_rng({SEED}), percentile 2.5/97.5",
                   "limits": "correlational; Whisper-small transcripts; one synthesis per sentence; not a causal test"},
        "sentence_level": {"n": len(sentences), "mean_phoneme_distance": float(d.mean()), "spearman": sp, "terciles": terciles},
        "word_level": {"sentences": len(keys), "words": len(words), **base,
                       "difference_in_differences": float(did(base)),
                       "difference_in_differences_bootstrap95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       "share_of_candidate_excess_errors_on_differing_words": float(excess_diff / (excess_diff + excess_same))},
        "sentences": sentences, "words": words,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    wl = out["word_level"]
    print(f"fed re-phonemization = manifest {fed_equals_manifest}/{len(man)}; "
          f"rho(distance, candidate WER) {sp['candidate_wer']['rho']:+.3f}, control {sp['control_wer']['rho']:+.3f}; "
          f"DiD {wl['difference_in_differences']:+.4f} {wl['difference_in_differences_bootstrap95']}; "
          f"share {wl['share_of_candidate_excess_errors_on_differing_words']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
