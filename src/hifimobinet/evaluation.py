"""Audit stored predictions and metrics; does not run TTS, ASR or UTMOSv2."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np


def wer_transform():
    import jiwer
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                          jiwer.ReduceToListOfListOfWords()])


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty JSON array of per-sentence observations")
    for row in rows:
        if not isinstance(row, dict) or row.get("error"):
            raise ValueError("Failed observations require an explicit missing-data protocol; refusing silent imputation")
        for key in ("rtf", "utmosv2", "wer"):
            if not isinstance(row.get(key), (int,float)) or not math.isfinite(row[key]):
                raise ValueError(f"Missing/non-finite {key}; refusing a silent change of denominator")
        if row['rtf'] < 0 or row['wer'] < 0:
            raise ValueError("RTF and WER cannot be negative")
    return rows


def summarize(rows: list[dict]) -> dict:
    import jiwer
    tf = wer_transform()
    refs=[r['text'] for r in rows]; hyps=[r['asr_text'] for r in rows]
    per=np.asarray([jiwer.wer(r,h,reference_transform=tf,hypothesis_transform=tf) for r,h in zip(refs,hyps)])
    corpus=jiwer.process_words(refs,hyps,reference_transform=tf,hypothesis_transform=tf)
    n_ref=corpus.hits+corpus.substitutions+corpus.deletions
    result={"classification":"new_audit_of_stored_observations", "n":len(rows),
            "mean_rtf":float(np.mean([r['rtf'] for r in rows])),
            "mean_utmosv2_predicted":float(np.mean([r['utmosv2'] for r in rows])),
            "macro_wer_stored":float(np.mean([r['wer'] for r in rows])),
            "macro_wer_recomputed":float(per.mean()),
            "max_abs_wer_difference":float(np.max(np.abs(per-[r['wer'] for r in rows]))),
            "corpus_wer_exploratory":corpus.wer,
            "word_counts":{"S":corpus.substitutions,"D":corpus.deletions,"I":corpus.insertions,"N":n_ref}}
    if all('synth_time_s' in r and 'audio_duration_s' in r for r in rows):
        durations=np.asarray([r['audio_duration_s'] for r in rows],dtype=float)
        times=np.asarray([r['synth_time_s'] for r in rows],dtype=float)
        if not (np.isfinite(durations).all() and (durations>0).all() and np.isfinite(times).all() and (times>=0).all()):
            raise ValueError("Invalid stored timing/duration")
        result['ratio_of_sums_rtf_exploratory']=float(times.sum()/durations.sum())
    return result


def check_pairs(left: list[dict], right: list[dict]) -> None:
    if len(left)!=len(right): raise ValueError("Paired analysis requires equal observation counts")
    # LJ indices alone do not establish a common sentence: compare IDs AND text.
    def identity(row):
        key=row.get('id',row.get('utt_id',row.get('idx')))
        if key is None or 'text' not in row: raise ValueError("Paired analysis requires IDs and text")
        return key,row['text']
    a=[identity(r) for r in left]; b=[identity(r) for r in right]
    if len(set(k for k,_ in a))!=len(a): raise ValueError("Duplicate observation IDs")
    if a!=b: raise ValueError("Paired analysis requires identical IDs, text and order; no automatic reordering")


def paired(left: list[dict], right: list[dict], metric: str) -> dict:
    from scipy.stats import wilcoxon
    check_pairs(left,right)
    x=np.asarray([r[metric] for r in left],float); y=np.asarray([r[metric] for r in right],float)
    # Match historical Harvard stats.py defaults; all-zero exception also preserved.
    test=wilcoxon(x,y,zero_method='wilcox',correction=False,alternative='two-sided',method='auto') if np.any(x!=y) else None
    return {'classification':'new_audit_of_stored_observations','metric':metric,'n_pairs':len(x),
            'mean_difference_left_minus_right':float(np.mean(x-y)),
            'wilcoxon_p':float(test.pvalue) if test else 1.0,
            'settings':{'zero_method':'wilcox','correction':False,'alternative':'two-sided','method':'auto'},
            'limits':'Exploratory paired sentence comparison; no equivalence margin; unadjusted p-value.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    summary=sub.add_parser('summarize'); summary.add_argument('results',type=Path)
    pair=sub.add_parser('paired'); pair.add_argument('left',type=Path); pair.add_argument('right',type=Path)
    pair.add_argument('--metric',choices=['rtf','utmosv2','wer'],required=True)
    args=parser.parse_args(argv)
    try:
        output=summarize(load_rows(args.results)) if args.command=='summarize' else paired(load_rows(args.left),load_rows(args.right),args.metric)
    except (ValueError,KeyError,OSError) as error:
        parser.exit(2,f'Evaluation refused: {error}\n')
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__': main()
