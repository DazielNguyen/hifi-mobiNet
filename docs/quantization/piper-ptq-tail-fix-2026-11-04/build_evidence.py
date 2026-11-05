"""Read existing experimental WAVs/traces; write only this report directory.
No inference, quantization, training, source modification, or network access.
Run with .local/ptq-ablation-20261104/venv/bin/python -B <this file> in WSL.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import csv
import hashlib
import json
import platform
import shutil
import wave
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import scipy

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
BASE = REPO / '.local/ptq-ablation-20261104'
MODELS = {'SeqIR': 'piper-seqir-no-vits2-cpn', 'ParIR': 'piper-parir-no-vits2-cpn', 'ResBlock2': 'piper-no-vits2-cpn'}
VARIANTS = ['fp32', 'old-int8', 'cumsum-only', 'common97']
LABELS = {'fp32': 'FP32 reference', 'old-int8': 'Old INT8: clipped duration', 'cumsum-only': 'Control: bypass ONLY CumSum FQ', 'common97': 'New INT8: common97'}
COLORS = {'fp32': '#285da8', 'old-int8': '#bb332b', 'cumsum-only': '#ad790b', 'common97': '#16784c'}
SR, NFFT, HOP = 22050, 1024, 256
SOURCES = {}
for name in ['audio', 'figures', 'evidence', 'evidence/code']:
    (OUT / name).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def register(path):
    path = Path(path).resolve()
    SOURCES[str(path.relative_to(REPO))] = sha(path)
    return path

def read(path):
    return json.loads(register(path).read_text(encoding='utf-8'))

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def copy(src, dst):
    register(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst)

def figure(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)

def load_wav(path):
    with wave.open(str(path), 'rb') as f:
        assert f.getframerate() == SR and f.getnchannels() == 1 and f.getsampwidth() == 2
        y = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(float) / 32768
    assert np.isfinite(y).all() and y.size > 0
    return y

def spectrum(y):
    frames = np.lib.stride_tricks.sliding_window_view(y, NFFT)[::HOP]
    window = signal.windows.hann(NFFT, sym=False)
    amp = np.abs(np.fft.rfft(frames * window, axis=1)) * 2 / window.sum()
    return {'f': np.fft.rfftfreq(NFFT, 1 / SR), 't': (np.arange(len(frames)) * HOP + NFFT / 2) / SR,
            'power': amp ** 2, 'db': 20 * np.log10(np.maximum(amp, 1e-6)),
            'rms': np.sqrt(np.mean(frames ** 2, axis=1))}

def region_metrics(y, spec, start, end):
    mask = (spec['t'] >= start) & (spec['t'] < end)
    active = mask & (20 * np.log10(np.maximum(spec['rms'], 1e-12)) > -50)
    band = (spec['f'] >= 80) & (spec['f'] <= 10000)
    hf = (spec['f'] >= 4000) & (spec['f'] <= 10000)
    p = spec['power'][active][:, band]
    flat = np.exp(np.mean(np.log(np.maximum(p, 1e-18)), axis=1)) / np.maximum(p.mean(axis=1), 1e-18)
    segment = y[round(start * SR):round(end * SR)]
    envelope = spec['rms'][mask]
    return {'start_s': start, 'end_s': end, 'rms_dbfs': float(20 * np.log10(max(np.sqrt(np.mean(segment ** 2)), 1e-12))),
            'median_spectral_flatness': float(np.median(flat)),
            'high_band_power_pct': float(100 * spec['power'][active][:, hf].sum() / max(p.sum(), 1e-18)),
            'centroid_hz': float((p.sum(axis=0) * spec['f'][band]).sum() / max(p.sum(), 1e-18)),
            'envelope_cv': float(envelope.std() / max(envelope.mean(), 1e-12)),
            'active_frames': int(active.sum()), 'total_frames': int(mask.sum())}

texts = read(BASE / 'texts.json')
asr = {}
for line in register(BASE / 'asr-scores.jsonl').read_text().splitlines():
    row = json.loads(line)
    asr[row['wav_path']] = row
common_alignment = read(BASE / 'common97-alignment-verification.json')
copy(BASE / 'common97-alignment-verification.json', OUT / 'evidence/common97-alignment-verification.json')
for f in ['summary.csv', 'PROPOSED-common97.json', 'environment-freeze.txt', 'asr-environment.json', 'graph-audit.json']:
    copy(BASE / f, OUT / 'evidence' / f)
code = {
    'old-openvino_suite.py': REPO / 'results/edge-n100-openvino/2026-11-01/evidence/openvino_suite.py',
    'generate-path-commons.py': REPO / 'vendor/edgetts/vits/commons.py',
    'inference-models.py': REPO / 'vendor/edgetts/vits/models.py',
    'new-run_experiment.py': BASE / 'run_experiment.py',
    'new-common97_validation.py': BASE / 'common97_validation.py',
    'control-cumsum_causal_probe.py': BASE / 'cumsum_causal_probe.py',
    'control-legacy_pair_probe.py': BASE / 'legacy_pair_probe.py',
    'verify_common97_alignment.py': BASE / 'verify_common97_alignment.py',
}
for name, path in code.items():
    copy(path, OUT / 'evidence/code' / name)

measurements, traces_all = [], {}
cache = {}
for label, model_id in MODELS.items():
    inventory = read(BASE / 'legacy-paired-v3' / model_id / 'int8/fq-inventory.json')
    dump(OUT / 'evidence' / f'{label}-old-cumsum-quantizer.json', next(r for r in inventory if r['name'] == '/CumSum/fq_output_0'))
    traces = {}
    for variant, dirname in [('old-int8', 'paired_original_trace'), ('cumsum-only', 'paired_bypass_cumsum')]:
        path = BASE / 'cumsum-causal-v1' / model_id / dirname / 'alignment-trace.json'
        traces[variant] = read(path)
        copy(path, OUT / 'evidence' / f'{label}-{variant}-alignment.json')
    traces_all[label] = traces
    for case in ['reported', 'stress_300']:
        old_trace = next(r for r in traces['old-int8'] if r['case'] == case and r['seed'] == 1234)
        for variant in VARIANTS:
            folder = {'fp32': BASE / 'legacy-paired-v3' / model_id / 'fp32',
                      'old-int8': BASE / 'cumsum-causal-v1' / model_id / 'paired_original_trace',
                      'cumsum-only': BASE / 'cumsum-causal-v1' / model_id / 'paired_bypass_cumsum',
                      'common97': BASE / 'common97-paired-v1' / model_id / 'common97'}[variant]
            src = folder / f'{case}-s1234.wav'
            if case == 'reported':
                copy(folder / 'record.json', OUT / 'evidence/run-records' / label / f'{variant}.json')
            dst = OUT / 'audio' / label / case / f'{variant}.wav'
            copy(src, dst)
            meta = next(r for r in read(folder / 'audio.json') if r['case'] == case and r['seed'] == 1234)
            assert sha(src) == meta['wav_sha256']
            score = asr[str(src.relative_to(BASE))]
            assert score['wav_sha256'] == sha(src)
            y = load_wav(dst)
            spec = spectrum(y)
            cache[label, case, variant] = (y, spec)
            row = {'model': label, 'model_id': model_id, 'case': case, 'variant': variant,
                   'seed': 1234, 'text': score['text'], 'wav': str(dst.relative_to(OUT)), 'wav_sha256': sha(dst),
                   'source_wav': str(src.relative_to(REPO)), 'seconds': len(y) / SR,
                   'samples': len(y), 'raw_peak_before_normalization': meta['raw_peak'],
                   'raw_rms_before_normalization': meta['raw_rms'],
                   'wer': score['wer'], 'hypothesis': score['hypothesis'], 'asr': score,
                   'old_fq_high_seconds': old_trace['fq_high_seconds']}
            if case == 'stress_300':
                row['regions'] = {name: region_metrics(y, spec, start, end) for name, start, end in [('head', 0.5, 5.5), ('tail', 8, 14)]}
            measurements.append(row)

    # Four WAVs on identical axes; magnitude colour scale is fixed for every model/variant.
    fig, axes = plt.subplots(4, 2, figsize=(15, 11.8), width_ratios=[1, 2.4], layout='constrained')
    case = 'stress_300'
    threshold = next(r for r in traces['old-int8'] if r['case'] == case and r['seed'] == 1234)['fq_high_seconds']
    max_t = max(len(cache[label, case, v][0]) / SR for v in VARIANTS)
    for i, v in enumerate(VARIANTS):
        y, s = cache[label, case, v]
        row = next(r for r in measurements if r['model'] == label and r['case'] == case and r['variant'] == v)
        axes[i, 0].plot(np.arange(0, len(y), 10) / SR, y[::10], lw=.35, color=COLORS[v])
        axes[i, 0].set(ylim=(-1.02, 1.02), xlim=(0, max_t), ylabel='PCM amplitude')
        axes[i, 0].set_title(f'{LABELS[v]}\n{row["seconds"]:.2f} s | ASR WER {row["wer"]:.1%}', fontsize=10, loc='left')
        im = axes[i, 1].pcolormesh(s['t'], s['f'] / 1000, s['db'].T, cmap='magma', vmin=-85, vmax=-5, shading='auto', rasterized=True)
        axes[i, 1].set(ylim=(0, 10), xlim=(0, max_t), ylabel='Frequency (kHz)')
        axes[i, 1].axvspan(8, 14, color='white', alpha=.07)
        if v == 'old-int8':
            for ax in axes[i]: ax.axvline(threshold, color='#00ffff', ls='--', lw=1.5)
            axes[i, 1].text(threshold + .15, 9.25, f'Old CumSum clamp ~{threshold:.2f} s', color='white', fontsize=9)
        if i == 3:
            for ax in axes[i]: ax.set_xlabel('Time (seconds); no time stretching')
    fig.colorbar(im, ax=axes[:, 1], label='STFT amplitude (dBFS), shared scale', shrink=.8)
    fig.suptitle(f'{label}: generated WAVs, same paragraph and paired noise (seed 1234)\nHann 1024 / hop 256 / 22050 Hz; each source WAV uses whole-utterance peak normalization', fontsize=13)
    figure(fig, f'{label}-waveform-spectrogram')

    # Tail close-up: harmonic/time structure, not an independently rescaled image.
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, layout='constrained')
    for ax, v in zip(axes, VARIANTS):
        _, s = cache[label, case, v]
        im = ax.pcolormesh(s['t'], s['f'] / 1000, s['db'].T, cmap='magma', vmin=-85, vmax=-5, shading='auto', rasterized=True)
        ax.set(xlim=(8, 12), ylim=(0, 4), ylabel='kHz', title=LABELS[v])
    axes[-1].set_xlabel('Time (seconds); same time window, phoneme timing can differ')
    fig.colorbar(im, ax=axes, label='STFT amplitude (dBFS)', shrink=.8)
    fig.suptitle(f'{label}: 8–12 second tail close-up from generated WAVs', fontsize=14)
    figure(fig, f'{label}-tail-detail')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for v in VARIANTS:
        y, s = cache[label, case, v]
        band = (s['f'] >= 80) & (s['f'] <= 10000)
        hf = (s['f'] >= 4000) & (s['f'] <= 10000)
        power = s['power'][:, band]
        flat = np.exp(np.mean(np.log(np.maximum(power, 1e-18)), axis=1)) / np.maximum(power.mean(axis=1), 1e-18)
        rms_db = 20 * np.log10(np.maximum(s['rms'], 1e-12))
        flat[rms_db <= -50] = np.nan
        axes[0, 0].plot(s['t'], rms_db, color=COLORS[v], label=v, lw=.9, alpha=.85)
        # Time-bin medians improve legibility without altering metric calculations.
        bins = np.arange(0, max_t, .25)
        med = [float(np.nanmedian(flat[(s['t'] >= t) & (s['t'] < t+.25)])) if np.isfinite(flat[(s['t'] >= t) & (s['t'] < t+.25)]).any() else np.nan for t in bins]
        axes[0, 1].plot(bins+.125, med, color=COLORS[v], label=v, lw=1.3)
        for j, (start, end, title) in enumerate([(0.5, 5.5, 'Head: 0.5–5.5 s'), (8, 14, 'Tail: 8–14 s')]):
            m = (s['t'] >= start) & (s['t'] < end) & (rms_db > -50)
            p = s['power'][m][:, band].mean(axis=0)
            p = p / max(p.sum(), 1e-18)
            axes[1, j].plot(s['f'][band]/1000, 10*np.log10(np.maximum(p, 1e-12)), label=v, color=COLORS[v], lw=1.2)
            axes[1, j].set(title=title, xlabel='Frequency (kHz)', ylabel='Normalized mean power (dB/bin)', ylim=(-70, 0))
    axes[0, 0].set(title='Short-time RMS envelope', xlabel='Time (s)', ylabel='RMS (dBFS)', ylim=(-65, 0))
    axes[0, 1].set(title='Spectral flatness (80 Hz–10 kHz)', xlabel='Time (s)', ylabel='Flatness: 0 tonal / 1 flat', ylim=(0, 1))
    for ax in axes[0]: ax.axvline(threshold, ls='--', color='gray'); ax.axvspan(8, 14, color='gray', alpha=.08)
    for ax in axes.ravel(): ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f'{label}: descriptors computed from WAVs; not intelligibility scores', fontsize=13)
    figure(fig, f'{label}-spectral-metrics')

# Reconstruct diagnostic curves using only stored trace values.
fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout='constrained')
alignment_rows = []
for i, (label, model_id) in enumerate(MODELS.items()):
    old = next(r for r in traces_all[label]['old-int8'] if r['case'] == 'stress_300' and r['seed'] == 1234)
    bypass = next(r for r in traces_all[label]['cumsum-only'] if r['case'] == 'stress_300' and r['seed'] == 1234)
    new = next(r for r in common_alignment if r['model'] == model_id and r['case'] == 'stress_300' and r['seed'] == 1234)
    axes[i, 0].plot(old['/CumSum']['values'], label='Before FQ', color=COLORS['fp32'])
    axes[i, 0].plot(old['/CumSum/fq_output_0']['values'], label='After old FQ', color=COLORS['old-int8'])
    axes[i, 0].axhline(old['fq_input_high'], ls=':', color=COLORS['old-int8'])
    axes[i, 0].set(title=f'{label}: cumulative duration', xlabel='Phoneme position', ylabel='Frames')
    for row, v in [(old, 'old-int8'), (bypass, 'cumsum-only')]:
        c = np.asarray(row['/Sub']['frame_coverage'])
        axes[i, 1].plot(np.arange(len(c))*HOP/SR, c, label=v, color=COLORS[v], lw=1.5)
    # Min=max=1 stored in independent validation; this is a verified constant series.
    assert new['min_alignment_column_sum'] == new['max_alignment_column_sum'] == 1
    axes[i, 1].plot(np.arange(new['audio_frames'])*HOP/SR, np.ones(new['audio_frames']), label='common97 (verified sum=1)', color=COLORS['common97'], ls='--')
    axes[i, 1].set(title=f'{label}: phoneme coverage per frame', xlabel='Time (s)', ylabel='Alignment column sum', ylim=(-.1, 1.25))
    for ax in axes[i]: ax.legend(fontsize=8); ax.grid(alpha=.2)
    alignment_rows.append({'model': label, 'fq_max_frames': old['fq_input_high'], 'fq_seconds': old['fq_high_seconds'],
                           'cumsum_required': old['/CumSum']['max'], 'old_zero_frames': old['/Sub']['zero_frames'],
                           'bypass_zero_frames': bypass['/Sub']['zero_frames'], 'new_zero_frames': new['zero_alignment_frames'],
                           'old_audio_seconds': old['seconds']})
fig.suptitle('Causal evidence from graph traces; waveform figures independently show the audible consequence', fontsize=13)
figure(fig, 'alignment-cause-all-models')

with register(BASE / 'summary.csv').open(encoding='utf-8-sig') as f:
    summaries = list(csv.DictReader(f))
quality = []
fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
for j, (variant, run, v) in enumerate([('FP32', 'full-v1', 'fp32'), ('Old INT8', 'legacy-paired-v3', 'int8'), ('CumSum only', 'cumsum-causal-v1', 'paired_bypass_cumsum'), ('New common97', 'common97-paired-v1', 'common97')]):
    vals = []
    for label, model_id in MODELS.items():
        row = next(r for r in summaries if r['model']==model_id and r['run']==run and r['variant']==v)
        val = float(row['corpus_wer']) * 100
        vals.append(val)
        quality.append({'model': label, 'variant': variant, 'corpus_wer_pct': val, 'audio_count': int(row['audio_count'])})
    bars = ax.bar(np.arange(3)+(j-1.5)*.19, vals, .19, label=variant, color=list(COLORS.values())[j])
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
ax.set(xticks=np.arange(3), xticklabels=list(MODELS), ylabel='Corpus WER (%)', ylim=(0, 49), title='Same six diagnostic texts × two noise seeds per model / variant')
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=.2)
figure(fig, 'quality-comparison')

# Retain user-supplied pair and the exact new samples the user just listened to.
for name in ['demo_fp32_seq.wav', 'demo_ptq.wav']:
    copy(BASE / 'user-references' / name, OUT / 'audio/user-reference' / name)
for name in ['SeqIR-INT8-common97-long.wav', 'ParIR-INT8-common97-long.wav', 'record.json']:
    copy(REPO / '.local/ptq-listening-20261104' / name, OUT / 'audio/listened-new' / name)
copy(BASE / 'user-wavs-asr.json', OUT / 'evidence/user-wavs-asr.json')
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
for ax, name, label in zip(axes, ['demo_fp32_seq.wav', 'demo_ptq.wav'], ['User-supplied SeqIR FP32', 'User-supplied SeqIR INT8']):
    y = load_wav(OUT / 'audio/user-reference' / name); s = spectrum(y)
    im = ax.pcolormesh(s['t'], s['f']/1000, s['db'].T, cmap='magma', vmin=-85, vmax=-5, shading='auto', rasterized=True)
    ax.set(title=f'{label} | {len(y)/SR:.4f} s', ylabel='Frequency (kHz)', ylim=(0, 10))
axes[-1].set_xlabel('Time (s)')
fig.colorbar(im, ax=axes, label='STFT amplitude (dBFS)')
fig.suptitle('Original user WAVs: unmatched RNG / no request metadata; observational comparison only', fontsize=12)
figure(fig, 'user-wavs-spectrogram')

dump(OUT / 'evidence/measurements.json', measurements)
dump(OUT / 'evidence/alignment-summary.json', alignment_rows)
dump(OUT / 'evidence/quality-summary.json', quality)
dump(OUT / 'evidence/analysis-method.json', {'sample_rate': SR, 'n_fft': NFFT, 'hop_length': HOP,
    'window': 'periodic Hann', 'magnitude_scale': '2*abs(rfft(windowed_frame))/sum(window)',
    'figure_db_range': [-85, -5], 'frequency_band_metrics_hz': [80, 10000], 'high_band_hz': [4000,10000],
    'active_frame_min_rms_dbfs': -50, 'head_seconds': [.5, 5.5], 'tail_seconds': [8,14],
    'no_time_warp_or_waveform_subtraction': True, 'source_wavs_individually_peak_normalized': True,
    'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__})
for rel, digest in SOURCES.items():
    assert sha(REPO / rel) == digest, f'Source changed: {rel}'
dump(OUT / 'evidence/source-hashes.json', [{'repo_relative_path': p, 'sha256': h, 'unchanged_after_analysis': True} for p,h in sorted(SOURCES.items())])
flat = []
for r in measurements:
    for region, m in r.get('regions', {}).items(): flat.append({'model':r['model'],'variant':r['variant'],'region':region,**m})
with (OUT / 'evidence/acoustic-metrics.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
print(json.dumps({'wav_comparisons': len(measurements), 'figures': len(list((OUT/'figures').glob('*.png'))), 'sources_verified':len(SOURCES)}))
for r in flat:
    if r['region']=='tail': print(json.dumps(r))
