# Idle-machine timing session, 2026-10-21

*Tiếng Việt: [README.vi.md](README.vi.md)*

These are the records of one CPU timing session on 2026-10-21, 09:05–09:09 +07. It ran after
the Piper-IR training sequence had completed and no trainer was left.

Three models were synthesized on Harvard-720 back to back, each in a fresh process of
`scripts/evaluation/harvard/synthesize.py`:
1. Piper_no_VITS2_cpn, the control (epoch 1403);
2. Piper-ParIR (epoch 1486);
3. Piper-SeqIR (epoch 1455).

The protocol was PyTorch FP32, 12 threads, normal priority, forward call timed, one pass per
sentence and no warm-up.

| Model | Mean RTF | Relative to control | Faster than control |
|---|---:|---:|---:|
| Piper_no_VITS2_cpn (control) | 0.0510 | — | — |
| Piper-ParIR | 0.0392 | −23.2% | 653 / 720 |
| Piper-SeqIR | 0.0340 | −33.3% | 690 / 720 |

- **Control:** its WAVs were byte-identical to the 2026-10-10 WAVs (720/720). Its RTF was 1.21× its own 2026-10-10 value, so RTF from this session is comparable only within the session.
- **ParIR:** its WAVs were identical to the 2026-10-16 evaluation (720/720).

The analysis, the limits and the quality results are in
[`../piper-seqir-no-vits2-cpn/README.md`](../piper-seqir-no-vits2-cpn/README.md), section 4.
The SeqIR synthesis record of this session is `../piper-seqir-no-vits2-cpn/harvard720_synth_record.json`.

| File | Content |
|---|---|
| `piper_no_vits2_cpn_ctrl_synth_record.json` | Control synthesis: hashes, protocol, environment, per-sentence timing and WAV SHA-256 |
| `piper_parir_timing_synth_record.json` | ParIR synthesis in this session, same fields |
| `session_control_parir.json`, `session_control_seqir.json` | Control reproducibility, session drift, paired RTF test against the control |
| `parir_determinism.json` | ParIR WAV hashes and durations against the 2026-10-16 evaluation |
| `machine_state_before.txt`, `machine_state_after.txt` | Load average, GPU use and top processes before and after the timing |
