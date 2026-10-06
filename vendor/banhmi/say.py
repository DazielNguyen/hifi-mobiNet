#!/usr/bin/env python3
"""Synthesize a sentence with an ONNX Piper-compatible voice.

Mirrors piper_train's say.py, but runs ONNX Runtime inference (no PyTorch
checkpoint needed) and uses banhmi_phonemize instead of piper_phonemize.

Usage:
    python -m banhmi_train.say \
        --model en_US-sam-medium.onnx \
        --text "Hello, how are you?" \
        --output test.wav
"""
import argparse
import logging

import numpy as np
import onnxruntime

from banhmi_phonemize import phonemize_espeak, phoneme_ids_espeak

from .audio_utils import audio_float_to_int16
from .mixed_lang import phonemize_mixed
from .wavfile import write as write_wav

_LOGGER = logging.getLogger("banhmi_train.say")


def text_to_phoneme_ids(
    text: str, language: str = "en-us", detect_vietnamese: bool = False
):
    if detect_vietnamese:
        sentences_phonemes = phonemize_mixed(text, en_voice=language)
    else:
        sentences_phonemes = phonemize_espeak(text, language)
    phonemes = [p for sentence in sentences_phonemes for p in sentence]
    return phoneme_ids_espeak(phonemes)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="banhmi_train.say")
    parser.add_argument("--model", required=True, help="Path to ONNX voice model")
    parser.add_argument("--text", required=True, help="Sentence to synthesize")
    parser.add_argument("--output", required=True, help="Path to write the .wav file")
    parser.add_argument("--language", default="en-us", help="espeak-ng voice/language")
    parser.add_argument(
        "--detect-vietnamese",
        action="store_true",
        help=(
            "Phonemize Vietnamese proper nouns embedded in the text with "
            "the Vietnamese voice (see banhmi_train.mixed_lang)"
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--speaker-id", type=int, default=None)
    args = parser.parse_args()

    phoneme_ids = text_to_phoneme_ids(
        args.text, args.language, args.detect_vietnamese
    )
    _LOGGER.info("Text: %s", args.text)
    _LOGGER.info("Phoneme ids (%d): %s", len(phoneme_ids), phoneme_ids)

    model = onnxruntime.InferenceSession(args.model)

    text_arr = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text_arr.shape[1]], dtype=np.int64)
    scales = np.array(
        [args.noise_scale, args.length_scale, args.noise_w], dtype=np.float32
    )

    model_inputs = {"input": text_arr, "input_lengths": text_lengths, "scales": scales}

    # Models exported by banhmi_train.export_onnx have no "sid" input at all
    # (this project is permanently single-speaker); Piper-exported voices
    # (e.g. a pretrained checkpoint) always declare one even when unused, so
    # only feed it when the graph actually has it.
    graph_input_names = {i.name for i in model.get_inputs()}
    if "sid" in graph_input_names:
        model_inputs["sid"] = (
            np.array([args.speaker_id], dtype=np.int64)
            if args.speaker_id is not None
            else None
        )

    audio = model.run(None, model_inputs)[0].squeeze((0, 1))
    audio = audio_float_to_int16(audio.squeeze())

    write_wav(args.output, args.sample_rate, audio)
    _LOGGER.info(
        "Wrote %s (%.2f sec)", args.output, audio.shape[-1] / args.sample_rate
    )


if __name__ == "__main__":
    main()
