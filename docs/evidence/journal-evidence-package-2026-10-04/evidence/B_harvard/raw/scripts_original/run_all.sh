#!/bin/bash
# Strictly sequential: one model at a time, synth then score, never overlapping.
set -e
S=<WSL_PROJECT_HOME>/scripts/harvard
PY=<WSL_PROJECT_HOME>/env/bin/python
PIPER_PY=<WSL_PROJECT_HOME>/piper_env/bin/python
cd <WSL_PROJECT_HOME>/BanhmiTTS

$PY $S/synth_banhmi.py baseline "<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt"
$PY $S/score.py baseline
$PY $S/synth_banhmi.py mrf "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt"
$PY $S/score.py mrf
$PY $S/synth_banhmi.py seq "<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt"
$PY $S/score.py seq
$PIPER_PY $S/synth_piper.py
$PY $S/score.py piper
echo ALL_DONE
