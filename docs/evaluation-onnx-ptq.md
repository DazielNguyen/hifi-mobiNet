# Protocol: ONNX export, PTQ and Harvard-720 evaluation of the three Config A models

Set up on 2026-10-21, before any graph was exported. The thresholds, scope and calibration set
below are fixed here. A failed check stops the round; nothing is relaxed afterwards.

## 1. Models

The selected checkpoints of the Piper-IR round, unchanged:

| Label | Model | Checkpoint | SHA-256 |
|---|---|---|---|
| `piper-no-vits2-cpn` | Piper_no_VITS2_cpn (ResBlock2) | `best-epoch=1403-val_loss_mel=19.7719.ckpt` | `2eeff535…6f0003` |
| `piper-parir-no-vits2-cpn` | Piper_ParIR_no_VITS2_cpn | `best-epoch=1486-val_loss_mel=19.0069.ckpt` | `d57f6abb…c97ee` |
| `piper-seqir-no-vits2-cpn` | Piper_SeqIR_no_VITS2_cpn | `best-epoch=1455-val_loss_mel=19.0332.ckpt` | `cc1ba70d…8c6637` |

Loading is the same as in `scripts/evaluation/harvard/synthesize.py`:
- `torch.load(weights_only=True)` and a strict state-dict load;
- the stored hyper-parameters must equal the training config;
- the decoder's weight norm is removed.

## 2. FP32 export

- **Interface.** The graph has the same interface as the Q05 graphs (`docs/evidence/claim-evidence-2026-10-07`, J-C011):
  - inputs `input` [batch, phonemes] int64, `input_lengths` [batch] int64 and `scales` [3] float32;
  - output `output` [batch, 1, samples] float32;
  - opset 18, `torch.onnx.export(dynamo=False)`, dynamic batch, phoneme and sample axes.
- **Export wrapper.** `scripts/evaluation/onnx/common.py:ExportWrapper` reproduces EdgeTTS `SynthesizerTrn.infer()` for the Config A path: single speaker, stochastic duration predictor, no F0, `max_len` None. It differs from `infer()` in two ways only:
  - The frame count is `y_lengths.max()` as a tensor. `infer()` calls `.item()`, and tracing would freeze that value to the trace input's length.
  - `torch.nan_to_num` is written with `isnan` and `±inf` comparisons. It has the same values and the same clamps.
  - A unit test checks, on a small randomly initialized model, that the wrapper and `infer()` give identical outputs.
- **Structure checks:**
  - exactly two `RandomNormalLike` nodes (duration noise under `/dp/`, acoustic noise);
  - no `enc_q` initializer, no quantized node, no FP16 initializer;
  - output rank 3 with a dynamic `samples` axis;
  - `onnx.checker` with `full_check`.
- **Matched-noise graph.** Built for validation and quality evaluation only; it is not a deployment artifact. The only change is that the two random nodes are replaced:
  - duration noise becomes an input with the random node's output name, [batch, 2, phonemes], as in Q05;
  - acoustic noise becomes a slice `[:, :, :frames]` of an input `noise_acoustic` [batch, 192, buffer frames], where `frames` is the frame count of the tensor the random node was shaped like.
  - A check confirms that removing the added inputs, slice nodes and their constants gives back the deployment graph without its random nodes.
- **Parity thresholds** (as Q05): NRMSE ≤ 1e-3 and max abs error ≤ 5e-3 against PyTorch with the same noise. The output shapes must be equal and all values finite.
  - **Required before anything else runs:** Harvard sentences idx 0–9 at the evaluation scales, plus idx 2 with a second seed (1235) and with length scales 0.8 and 1.2. That is 13 cases per model.
  - **Reported, not a gate:** parity of all 720 Harvard sentences, measured during the quality synthesis (section 4).
- **Native-random checks** (as Q05):
  - three sentences through the deployment graph give finite, non-silent audio;
  - the ORT RNG behaviour is recorded: same seed in fresh sessions, and consecutive calls in one session.

## 3. PTQ

- **Recipe.** The same recipe as the historical INT8 graphs of baseline and MRF (J-C024; `scripts/historical/ptq_quantize.py` in the evidence package):
  - ONNX Runtime `quantize_static`, QOperator format;
  - weights QInt8 per channel, activations QUInt8;
  - MinMax calibration (the default).
- **Scope:**
  - `Conv` nodes under `/enc_p/`, `/dp/` and `/flow/`;
  - every other node stays FP32, including the whole decoder `/dec/`;
  - the same scope for all three models. The wider historical SEQ scope (also `MatMul`) is not repeated.
- **Calibration** (fixes the known flaw of the historical set, J-C010: 3 of its 60 sentences were in the test split):
  - The set is the 60-sentence candidate designed on 2026-10-04 and never used: `docs/evidence/claim-evidence-2026-10-07/source-records/calibration/calibration60.candidate.jsonl`.
  - Checked before quantizing: all 60 IDs are in the canonical train split, none is in validation or test, and the phoneme IDs equal the training `dataset.jsonl` rows.
  - Sentences are fed in file order at scales 0.667/1.0/0.8, after `onnxruntime.set_seed(1234)`.
- **Checks:**
  - number of quantized Conv nodes by module, and zero quantized nodes under `/dec/`;
  - every initializer used by `/dec/` nodes is byte-identical to the FP32 parent;
  - the two `RandomNormalLike` nodes are kept;
  - a second quantization from the same inputs gives a byte-identical graph.
- A matched-noise INT8 graph is derived from the INT8 deployment graph by the same replacement as in section 2.

## 4. Quality evaluation (Harvard-720)

- **Graphs.** Six matched-noise graphs: FP32 and INT8 for each model. ORT CPU, intra-op 2 threads, inter-op 1, sequential, as Q05.
- **Noise, per sentence:**
  1. `torch.manual_seed(1234)`, then the PyTorch forward exactly as `harvard/synthesize.py`, capturing its duration noise (`randn`) and acoustic noise (`randn_like`).
  2. The acoustic buffer is that noise followed by `numpy.random.default_rng(1234 + idx)` normal values (as many frames again plus 200), for graphs that predict more frames. A graph that needs more frames than the buffer stops the run.
  3. FP32-ONNX, INT8 and PyTorch therefore share the same noise. Differences between them come from export and quantization, not from different random draws.
- **Recorded per sentence:**
  - the ONNX WAV, written as in `harvard/synthesize.py` (peak-normalized int16, 22,050 Hz);
  - whether the PyTorch WAV of this pass is byte-identical to the stored PyTorch evaluation WAV (a determinism check; for SeqIR, the missing second synthesis);
  - waveform parity against PyTorch: equal length, NRMSE, max abs error.
  - The timing fields of this pass are not used, because the PyTorch reference runs in the same loop.
- **Scoring:** `harvard/score.py` unchanged (UTMOSv2 `fusion_stage3` fold 0, Whisper-small, JiWER). All six sets are scored in one session.
- **Comparisons:** `harvard/compare.py`, paired by sentence, on UTMOSv2 and WER, per model:
  - **INT8 vs FP32-ONNX** (the quantization effect; `--no-historical`);
  - **FP32-ONNX vs PyTorch,** using the stored PyTorch results (the export effect; `--no-historical`);
  - **INT8 vs the four historical systems,** for context.

## 5. Speed (idle machine)

- **When:** after the quality synthesis and before scoring, with no training running and the machine state recorded before and after.
- **Graphs:** the six deployment graphs (native random noise, `onnxruntime.set_seed(1234)` once per session), ORT CPU, sequential, inter-op 1.
  - Intra-op 2 threads: the Q05 and historical ONNX setting.
  - Intra-op 12 threads: the same thread count as the PyTorch timing.
- **Order:** round A (no_VITS2 FP32, INT8, ParIR FP32, INT8, SeqIR FP32, INT8), then round B in reverse order, for each thread setting.
- **Control:** a PyTorch `harvard/synthesize.py` run of Piper_no_VITS2_cpn opens the session. It links the session to the 2026-10-21 PyTorch timing.
- **Protocol:** one timing per sentence, no warm-up, timed region `session.run` only (as before). RTF is compared only within this session.
- **Size:** graph file bytes for each graph.

## 6. Rules

- Checkpoints, configs and the scorer are not changed. Harvard-720 is not used to choose anything: the scope and the calibration set are fixed above.
- One round, no sweep. If a check fails, the failure is reported and the round stops.
- Graphs, WAVs and logs stay outside Git, under `training_output/onnx-ptq/`.
- Records are copied to `results/piper-ir-onnx-ptq/`, with README.md and README.vi.md, and listed in `results/manifest.json`.

## 7. Deviation found after the round (recorded 2026-10-22)

Section 1 says the decoder's weight norm is removed, as in `harvard/synthesize.py`. The flow's weight norm was not removed.
- **Effect on the graphs:** EdgeTTS's flow coupling layers (`WN`) keep their weight norm, so their 32 convolution weights are computed in the graph at every run.
- **Effect on the recipe of section 3:**
  - `quantize_static` could not quantize those weights as constants. It quantized them at run time, per tensor, as UINT8.
  - The declared recipe (static QInt8 per channel) therefore holds for only 8 of the 40 flow convolutions, and for all of enc_p and dp.
  - The INT8 graphs keep 27.8 MB of FP32 flow weights.
- **Not affected:** none of the declared checks failed, because none of them tested this.
- **Correction for a repeat round** (implemented; run as round 2 on 2026-10-26, every check passed, records in `results/piper-ir-onnx-ptq-v2/`; `docs/plan-onnx-ptq-v2.md`): call `remove_weight_norm()` on each flow coupling layer's `WN` before export, and add checks that every in-scope QLinearConv has a constant per-channel INT8 weight and that no `ReduceL2` (weight-norm) node remains.
- **Records:** the current results stay as they are, labelled with this deviation (`results/README.md`, `results/piper-ir-onnx-ptq/README.md`).
