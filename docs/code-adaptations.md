# Code adaptations

1. `vendor/` preserves unchanged reference files. Their original SHA-256 values
   appear in the source map and were committed before executable adaptations.
2. `src/hifimobinet/architecture/vits/` relocates current Banhmi model components.
   Forward and inference arithmetic are retained. An unsupported Vocos import is
   replaced by a class that raises explicitly on construction. The MAS Cython
   import is deferred until the training-only function is called. Training itself
   is not exposed or validated; do not use this as a replacement training recipe.
3. The supported inference entry point consumes identified, pre-existing Q05
   ONNX graphs. It does not export the relocated PyTorch classes or deserialize
   Lightning checkpoints. No equivalence to the historical quality-evaluation
   runtime is claimed.
4. Text encoding calls the original Banhmi phonemizer (`en-us`) and flattens its
   sentence phonemes before the original ID mapping. No replacement normalizer,
   punctuation policy or blank-insertion algorithm is introduced.
5. Newly generated WAVs use the existing peak-to-PCM16 conversion. Historical
   comparison WAVs are returned byte for byte with no gain/loudness processing.
6. Two ORT intra-op threads and one inter-op thread are demo resource settings,
   not a reconstruction of paper timing conditions. No demo timing is reported.

Some imported current code includes optional F0 or Vocos branches. Their presence
is not evidence of their use by the selected checkpoints. Original preprocessing
is reference-only and includes later F0 caching; no new preprocessing run occurs.
