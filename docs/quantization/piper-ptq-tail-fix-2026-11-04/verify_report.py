"""Verify report sources, copied WAVs, images, local links and complete SHA-256 manifest."""
import hashlib
import json
import re
import wave
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads((ROOT/p).read_text(encoding='utf-8'))
sources = read('evidence/source-hashes.json')
for row in sources:
    assert sha(REPO/row['repo_relative_path']) == row['sha256'], row['repo_relative_path']
measurements = read('evidence/measurements.json')
assert len(measurements) == 24
for row in measurements:
    assert sha(ROOT/row['wav']) == row['wav_sha256'] == sha(REPO/row['source_wav'])
    assert row['asr']['wav_sha256'] == row['wav_sha256']
    with wave.open(str(ROOT/row['wav']), 'rb') as f:
        assert f.getframerate() == 22050 and f.getsampwidth() == 2 and f.getnchannels() == 1
        assert f.getnframes() == row['samples']
        assert abs(f.getnframes()/22050-row['seconds']) < 1e-10
for row in read('audio/listened-new/record.json')['models']:
    assert sha(ROOT/'audio/listened-new'/row['wav']) == row['wav_sha256']
for row in read('evidence/user-wavs-asr.json'):
    assert sha(ROOT/'audio/user-reference'/row['file']) == row['sha256']
pngs = sorted((ROOT/'figures').glob('*.png'))
assert len(pngs) == 12 and len(list((ROOT/'figures').glob('*.svg'))) == 12
for path in pngs:
    with Image.open(path) as image:
        assert image.width > 1000 and image.height > 600
        image.verify()
link_targets = []
class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src'): link_targets.append(value)
for filename in ['BAO-CAO.html','NGHE-DOI-CHIEU.html']:
    parser=Links();parser.feed((ROOT/filename).read_text(encoding='utf-8'))
md = (ROOT/'README.vi.md').read_text(encoding='utf-8')
assert not re.search(r'__[A-Z_]+__', md)
link_targets += re.findall(r'\[[^\]]*\]\(([^)]+)\)',md)
checked = set()
for target in link_targets:
    url = urlparse(target)
    if url.scheme or not url.path: continue
    p = (ROOT/unquote(url.path)).resolve()
    assert p.is_relative_to(ROOT), f'Unexpected external file dependency: {p}'
    if p.name not in ['VERIFICATION.json','SHA256SUMS']: assert p.is_file(), f'Missing link: {target}'
    checked.add(str(p.relative_to(ROOT)))
report = {'utc':datetime.now(timezone.utc).isoformat(), 'checks':{
    'source_hashes_unchanged':len(sources), 'paired_wavs_bytes_headers_and_asr_hashes':24,
    'other_wavs_hashes':4, 'pngs_valid':12, 'svg_count':12, 'local_link_targets':len(checked),
    'unresolved_placeholders':0}, 'runtime_original_code_modified':False,
    'note':'Generated figures and file links verified. No automated browser playback test. No new ASR or N100 timing in this report task.'}
(ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
(ROOT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf-8')
for line in (ROOT/'SHA256SUMS').read_text().splitlines():
    digest,rel=line.split('  ',1)
    assert sha(ROOT/rel)==digest, rel
print(json.dumps({'status':'verified','manifest_files':len(files),'bytes':sum(p.stat().st_size for p in files),**report['checks']}))
