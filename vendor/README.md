# Unmodified reference imports

These are selective, byte-identical copies, inventoried in `docs/source-map.json`.
They are deliberately outside the installed package. Incomplete optional training
dependencies and original experimental branches are not supported entry points.

`banhmi/` is current source, not proven training-time source. Its original
preprocessing worker now caches F0 unconditionally; that current implementation
does not establish that the selected historical baseline/MRF/SEQ used F0. The
selected model configurations have F0 and Vocos disabled.

`piper/` preserves the external reference model implementation; its Lightning
training loader is not used to deserialize checkpoints in this release preparation.
`banhmi-phonemize/` preserves the frontend's native source, ID table and notices.
No cached build, compiled binary, environment or weights were copied.

Training imports (2026-10-06): `piper/vits/monotonic_align/`, `piper/__main__.py`,
`piper/build_monotonic_align.sh` and `piper/requirements.txt` come from Git
objects at the pinned Piper commit; `banhmi/train.py`, `banhmi/vits/training.py`,
`banhmi/vits/dataset.py`, `banhmi/vits/length_bucket_sampler.py` and
`banhmi/warm_length_cache.py` come from the BanhmiTTS working tree (unchanged
from its HEAD). They stay unmodified: `src/hifimobinet/training/` loads them by
path (`training/vendor.py`) or adapts copies in its own namespace. The compiled
MAS module is built into the ignored `piper/vits/monotonic_align/monotonic_align/`
directory by `scripts/training/build_mas.sh`. See `docs/training/`.
