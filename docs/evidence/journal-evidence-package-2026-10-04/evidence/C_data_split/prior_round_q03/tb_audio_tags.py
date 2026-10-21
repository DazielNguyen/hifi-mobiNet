"""Extracts, from TensorBoard event files written DURING training, the tags of
the logged audio samples. VitsModel._log_audio_samples tags each sample with
the utterance text of the first num_audio_samples validation utterances
(Subset(val_dataset, range(5))), so the tag sequence is a contemporaneous
record of the head of the validation split, per run phase.

Read-only and streaming: for every TFRecord it reads the 12-byte header; for
large records (audio) it reads only the first 2 KiB and decodes the protobuf
prefix by hand (Event.step -> Summary.Value.tag -> Audio.sample_rate /
length_frames all precede the encoded WAV blob), then seeks past the rest.
CRCs are not verified (files are read, not trusted for content beyond tags).
Usage: tb_audio_tags.py <out.json> <event files...>
"""
import json
import os
import struct
import sys

LARGE = 4096
PREFIX = 2048


def varint(b, i):
    shift = result = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        if not c & 0x80:
            return result, i
        shift += 7


def fields(b, i, end):
    """Yields (field_no, wire_type, value_or_(start,len)) until end/truncation."""
    while i < end:
        key, i = varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
            yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i + 8]
            i += 8
        elif wt == 5:
            yield fn, wt, b[i:i + 4]
            i += 4
        elif wt == 2:
            n, i = varint(b, i)
            yield fn, wt, (i, n)
            i += n
        else:
            return


def parse_prefix(b):
    step, out = None, []
    try:
        for fn, wt, v in fields(b, 0, len(b)):
            if fn == 1 and wt == 1:  # Event.wall_time (add_audio here passes no global_step)
                step = struct.unpack("<d", v)[0]
            elif fn == 5 and wt == 2:  # Event.summary
                s0, sn = v
                for f2, w2, v2 in fields(b, s0, min(s0 + sn, len(b))):
                    if f2 == 1 and w2 == 2:  # Summary.value
                        v0, vn = v2
                        tag, sr, frames = None, None, None
                        for f3, w3, v3 in fields(b, v0, min(v0 + vn, len(b))):
                            if f3 == 1 and w3 == 2:
                                tag = b[v3[0]:v3[0] + v3[1]].decode("utf-8", "replace")
                            elif f3 == 6 and w3 == 2:  # Value.audio
                                a0, an = v3
                                for f4, w4, v4 in fields(b, a0, min(a0 + an, len(b))):
                                    if f4 == 1 and w4 == 5:
                                        sr = struct.unpack("<f", v4)[0]
                                    elif f4 == 3 and w4 == 0:
                                        frames = v4
                                    elif f4 == 4:
                                        break
                        if tag is not None and frames is not None:
                            out.append((tag, frames / sr if sr else None))
    except (IndexError, struct.error):
        pass
    return step, out


result = {}
for path in sys.argv[2:]:
    recs, audio = 0, []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            (n,) = struct.unpack("<Q", hdr[:8])
            recs += 1
            if n < LARGE:
                f.seek(n + 4, 1)
                continue
            b = f.read(min(n, PREFIX))
            f.seek(n - len(b) + 4, 1)
            step, vals = parse_prefix(b)
            for tag, dur in vals:
                audio.append((step, tag, dur))
    # add_audio is called without global_step, so group in file order: one
    # validation epoch = one run of consecutive audio events, and a new group
    # starts whenever a tag repeats inside the current one.
    groups = []
    for wt, tag, dur in audio:
        if not groups or tag in [t for t, _, _ in groups[-1]]:
            groups.append([])
        groups[-1].append((tag, round(dur, 3) if dur else None, wt))
    seqs = {}
    for g in groups:
        key = json.dumps([t for t, _, _ in g], ensure_ascii=False)
        seqs.setdefault(key, []).append(g)
    iso = lambda t: __import__("datetime").datetime.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")
    result[path] = {
        "bytes": size, "records": recs, "audio_events": len(audio),
        "validation_logs": len(groups),
        "distinct_tag_sequences": [
            {"tags": json.loads(k), "n_logs": len(v),
             "first_wall_time_utc": iso(v[0][0][2]), "last_wall_time_utc": iso(v[-1][0][2]),
             "durations_first": [d for _, d, _ in v[0]], "durations_last": [d for _, d, _ in v[-1]]}
            for k, v in seqs.items()],
    }
    print(f"{path}: {recs} records, {len(groups)} validation logs, "
          f"{len(seqs)} distinct tag sequence(s)", flush=True)

json.dump(result, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
