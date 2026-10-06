# Notice

`banhmi_phonemize`'s C++/Python source in this directory is an independent
implementation, written for BanhmiTTS -- it does not reuse
[piper_phonemize](https://github.com/rhasspy/piper-phonemize)'s (Copyright
(c) 2023 Michael Hansen, MIT license) source code.

Its output-compatible symbol/id table (`DEFAULT_PHONEME_ID_MAP` in
`banhmi_phonemize/__init__.py`) and its punctuation/sentence-boundary
reconstruction approach (looping `espeak_TextToPhonemesWithTerminator` and
re-inserting punctuation from its terminator code) were arrived at by
studying piper_phonemize's approach, so that this project's phoneme output
and phoneme ids stay numerically compatible with it and with existing
Piper-trained models. `LICENSE-MIT-piper_phonemize.txt` is kept alongside
this package as a courtesy attribution for that, even though no piper_phonemize
source is compiled into it.

Phonemization itself is done by [espeak-ng](https://github.com/rhasspy/espeak-ng)
(a fork of https://github.com/espeak-ng/espeak-ng maintained by the Piper
project, adding the `espeak_TextToPhonemesWithTerminator` function this
package's C++ layer calls directly), compiled from source at build time.
espeak-ng is licensed under the **GNU General Public License v3.0** -- see
`espeak-ng.GPL-3.0.txt` in this package, or
https://github.com/rhasspy/espeak-ng/blob/master/COPYING for the full text.
