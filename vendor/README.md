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
