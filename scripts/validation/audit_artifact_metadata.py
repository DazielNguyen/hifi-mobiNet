"""Inspect model artifact metadata without loading checkpoint objects.

Only counts and value fingerprints enter the report. Literal findings stay private.
This heuristic scan is not a security or legal clearance.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pickletools
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

PATTERNS = {
    'host_path': re.compile(r'(?:[A-Za-z]:[\\/](?:Users|Documents)[\\/]|/(?:home|Users)/[^\s\x00\"\']+)', re.I),
    'credential': re.compile(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
    'email': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
}
STATE_KEYS = ('state_dict', 'optimizer_states', 'lr_schedulers', 'hyper_parameters',
              'callbacks', 'epoch', 'global_step', 'pytorch-lightning_version')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scan_strings(strings):
    hits = {category: set() for category in PATTERNS}
    for value in strings:
        for category, pattern in PATTERNS.items():
            hits[category].update(pattern.findall(value))
    return {category: {
        'unique_literal_count': len(values),
        'value_sha256': sorted(hashlib.sha256(value.encode()).hexdigest() for value in values),
    } for category, values in hits.items()}


def checkpoint_metadata(path):
    if not zipfile.is_zipfile(path):
        raise ValueError('Non-ZIP checkpoint requires a separate static review')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        pickles = [entry for entry in members if PurePosixPath(entry.filename).name == 'data.pkl']
        if len(pickles) != 1:
            raise ValueError('Expected exactly one checkpoint data.pkl member')
        member = pickles[0]
        if member.file_size > 16 * 1024 * 1024:
            raise ValueError('Checkpoint metadata exceeds the static parsing limit')
        raw = archive.read(member)
        strings = []
        opcode_count = 0
        for opcode, argument, position in pickletools.genops(raw):
            opcode_count += 1
            if isinstance(argument, str):
                strings.append(argument)
        return {
            'inspection': 'ZIP data.pkl opcode and literal-string inspection; no deserialization',
            'archive_member_count': len(members),
            'metadata_bytes': len(raw),
            'metadata_sha256': hashlib.sha256(raw).hexdigest(),
            'opcode_count': opcode_count,
            'literal_string_count': len(strings),
            'state_key_literal_presence': {key: key in strings for key in STATE_KEYS},
            'findings': scan_strings(strings),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--upload-record', type=Path, required=True)
    parser.add_argument('--audio-catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-metadata-directory', type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    manifest = json.loads((candidate / 'release-manifest.json').read_text())
    upload = json.loads(args.upload_record.read_text())
    remote_inventory = {row['relative_path']: row for row in upload['final_inventory']}
    catalog = json.loads(args.audio_catalog.read_text())
    sentences = {sample['text'].casefold() for sample in catalog['samples'] if sample.get('text')}
    rows = []
    for expected in manifest['files']:
        relative = PurePosixPath(expected['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Candidate manifest contains an unsafe path')
        path = candidate.joinpath(*relative.parts)
        if path.is_symlink() or candidate not in path.resolve().parents:
            raise ValueError('Candidate path escapes the reviewed directory')
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError('Candidate no longer matches its preparation inventory')
        remote = remote_inventory.get(str(relative))
        row = {'relative_path': str(relative), 'bytes': size, 'sha256': digest,
               'matches_verified_private_upload_identity': bool(remote and remote['sha256'] == digest and remote['bytes'] == size)}
        if path.suffix == '.ckpt':
            row.update(checkpoint_metadata(path))
        else:
            text = path.read_bytes().decode('utf-8', errors='replace')
            row.update({'inspection': 'UTF-8 replacement decoding and pattern scan; ONNX is not executed',
                        'findings': scan_strings([text])})
            if relative.parts[:2] == ('logs', 'public'):
                row['exact_case_insensitive_Harvard_sentence_matches'] = sum(sentence in text.casefold() for sentence in sentences)
        rows.append(row)
    remote_metadata = []
    if args.remote_metadata_directory:
        for name in ('README.md', 'docs/artifact-licensing.json', 'UPLOAD-STATUS.json', 'release-manifest.json'):
            path = args.remote_metadata_directory / name
            remote = remote_inventory[name]
            digest = sha256_file(path)
            if path.is_symlink() or digest != remote['sha256'] or path.stat().st_size != remote['bytes']:
                raise ValueError('Pinned remote metadata differs from the upload record')
            remote_metadata.append({
                'relative_path': name, 'bytes': remote['bytes'], 'sha256': digest,
                'findings': scan_strings([path.read_bytes().decode('utf-8', errors='replace')]),
            })
    covered = {row['relative_path'] for row in rows if row['matches_verified_private_upload_identity']}
    covered.update(row['relative_path'] for row in remote_metadata)
    result = {
        'schema_version': 1,
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python_version': sys.version.split()[0],
        'method': 'Python standard-library zipfile, pickletools, re and hashlib; no torch.load or pickle.load',
        'scope': 'Frozen model candidate plus optional pinned current remote metadata; no checkpoint objects are loaded.',
        'private_model_revision_basis': upload['model_revision'],
        'files': rows,
        'current_remote_metadata': remote_metadata,
        'verified_private_inventory_covered': len(covered),
        'verified_private_inventory_total': len(remote_inventory),
        'Harvard_comparison': {'catalog_sentence_count': len(sentences), 'comparison': 'Exact complete-string substring match after case folding; no normalization beyond case folding.'},
        'limits': [
            'Patterns cover selected literal host paths, token shapes and email formats, not every private value or secret.',
            'Pickle inspection does not reconstruct the object graph, execute callbacks or inspect tensor-storage contents.',
            'State-key literal presence does not prove every value or a complete training recipe.',
            'ONNX scanning examines decoded bytes, not full graph semantics or encoded/compressed strings.',
            'No sentence match does not exclude partial, normalized, encoded or other copyrighted text.',
            'Current metadata and file identities do not prove source bytes at historical runtime.',
            'This report does not establish legal clearance or safe checkpoint deserialization.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    summary = [{'file': row['relative_path'], 'counts': {name: value['unique_literal_count'] for name, value in row['findings'].items()}}
               for row in rows if any(value['unique_literal_count'] for value in row['findings'].values())]
    print(json.dumps({'files_scanned': len(rows), 'findings': summary}))


if __name__ == '__main__':
    main()
