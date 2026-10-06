from copy import deepcopy
from pathlib import Path
import json
import pytest
from hifimobinet.evaluation import check_pairs,load_rows,summarize,paired

FIXTURE=Path(__file__).parent/'fixtures/metrics.json'


def test_known_word_counts_and_distinct_rtf_aggregations():
    out=summarize(load_rows(FIXTURE))
    assert out['n']==2
    assert out['word_counts']=={'S':1,'D':0,'I':0,'N':6}
    assert out['macro_wer_recomputed']==pytest.approx(1/6)
    assert out['max_abs_wer_difference']==0
    assert out['mean_rtf']==pytest.approx(.2)
    assert out['ratio_of_sums_rtf_exploratory']==pytest.approx(.25)
    assert out['mean_utmosv2_predicted']==3.5


def test_pairing_rejects_different_sentences_and_order():
    rows=load_rows(FIXTURE)
    assert paired(rows,rows,'wer')['wilcoxon_p']==1
    with pytest.raises(ValueError): check_pairs(rows,list(reversed(rows)))
    other=deepcopy(rows); other[0]['text']='A different sentence'
    with pytest.raises(ValueError): check_pairs(rows,other)
    other=deepcopy(rows); other[1]['idx']=0
    with pytest.raises(ValueError): check_pairs(other,other)


def test_finite_observations_required(tmp_path):
    rows=load_rows(FIXTURE); rows[0]['wer']=None
    p=tmp_path/'bad.json'; p.write_text(json.dumps(rows))
    with pytest.raises(ValueError): load_rows(p)


def test_known_wilcoxon_direction():
    # Five distinct positive differences: exact two-sided p = 2 / 2**5.
    left=[{'idx':i,'text':str(i),'wer':float(i+1)} for i in range(5)]
    right=[{'idx':i,'text':str(i),'wer':0.0} for i in range(5)]
    out=paired(left,right,'wer')
    assert out['wilcoxon_p']==pytest.approx(.0625)
    assert out['mean_difference_left_minus_right']==3
