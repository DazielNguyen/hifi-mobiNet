"""VITS (Conditional VAE + GAN, Kim et al. 2021) model, adapted for
BanhmiTTS's fixed single-speaker LJSpeech scenario.

Split into one file per architectural component, rather than the two
large models.py/modules.py files upstream uses, for easier per-piece
debugging:

- utils/    -- shared low-level building blocks (math, attention, WaveNet
               conditioner, normalizing-flow steps, monotonic alignment
               search).
- modules/  -- the named components built from those (TextEncoder,
               PosteriorEncoder, Generator, the discriminators, the
               duration predictor/discriminator, SynthesizerTrn).
- dataset.py / losses.py -- training-pipeline level, not part of the
               model itself.
"""
