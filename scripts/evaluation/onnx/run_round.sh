#!/bin/bash
# ONNX export, PTQ and Harvard-720 evaluation of the three Config A models
# (protocol: docs/evaluation-onnx-ptq.md). Run from the training clone root:
#
#   SPY=<python with torch+onnx+onnxruntime+scorers> TPY=<training venv python> \
#   RESULTS=<repository>/results/piper-ir-onnx-ptq bash <repository>/scripts/evaluation/onnx/run_round.sh
#
# Stages leave markers in $OUT/stages, so a re-run resumes after the last finished stage.
# A failed check stops the round (set -e on every stage command). Nothing here trains,
# commits or deletes. Graphs, WAVs and logs stay in $OUT (outside Git); records are copied
# to $RESULTS at the end.
set -uo pipefail
: "${SPY:?set SPY}" "${TPY:?set TPY}" "${RESULTS:?set RESULTS}"
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)          # scripts/evaluation/onnx
REPO=$(cd "$SD/../../.." && pwd)
H=$REPO/scripts/evaluation/harvard
OUT=${OUT:-training_output/onnx-ptq}
DATASET=${DATASET:-data/ljspeech-medium/training_output/ljspeech/medium/dataset.jsonl}
export CUDA_VISIBLE_DEVICES=""
LABELS=(piper-no-vits2-cpn piper-parir-no-vits2-cpn piper-seqir-no-vits2-cpn)
declare -A CKPT=([piper-no-vits2-cpn]="best-epoch=1403-val_loss_mel=19.7719.ckpt"
                 [piper-parir-no-vits2-cpn]="best-epoch=1486-val_loss_mel=19.0069.ckpt"
                 [piper-seqir-no-vits2-cpn]="best-epoch=1455-val_loss_mel=19.0332.ckpt")
mkdir -p "$OUT/stages"
LOG=$OUT/round.log
log() { echo "[$(date '+%F %T %z')] $*" | tee -a "$LOG"; }
have() { [ -e "$OUT/stages/$1" ]; }
done_() { touch "$OUT/stages/$1"; log "STAGE $1 done"; }
fail() { log "FAILED: $1"; [ -n "${NOTIFY_USER:-}" ] && /mnt/c/Windows/System32/msg.exe "$NOTIFY_USER" "onnx-ptq FAILED: $1" 2>/dev/null; exit 1; }
py() { "$SPY" -W ignore "$@" >>"$LOG" 2>&1; }
id_of() { echo "${1//-/_}"; }                              # piper-no-vits2-cpn -> piper_no_vits2_cpn
ckpt() { echo "training_output/$1/checkpoints/${CKPT[$1]}"; }
# trainer / controller processes: argv[0] must be a python interpreter (a tmux server keeps the
# command it was started with in its own argv and must not count)
busy() { pgrep -f '^[^ ]*python[0-9.]* (-m hifimobinet\.training\.train|[^ ]*run_sequence\.py)( |$)' >/dev/null; }
state() { { date '+%F %T %z'; uptime; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader;
            ps -eo pcpu,comm --sort=-pcpu | head -6; } > "$1" 2>&1; }

busy && fail "a training process is running"
log "round start; scripts from $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (modified files)')"

# 1. export and 2. PTQ (gates)
for L in "${LABELS[@]}"; do
  have "export_$L" || { py "$SD/export.py" --label "$L" --config "configs/training/$L.yaml" --checkpoint "$(ckpt "$L")" \
                          --out-dir "$OUT/$L" || fail "export $L"; done_ "export_$L"; }
  have "quantize_$L" || { py "$SD/quantize.py" --label "$L" --fp32 "$OUT/$L/${L}_fp32.onnx" --dataset "$DATASET" \
                            --out-dir "$OUT/$L" || fail "quantize $L"; done_ "quantize_$L"; }
done

# 3. quality synthesis through the matched-noise graphs
for L in "${LABELS[@]}"; do for P in fp32 int8; do
  have "synth_${L}_$P" || { py "$SD/synthesize.py" --precision "$P" --config "configs/training/$L.yaml" --checkpoint "$(ckpt "$L")" \
      --graph "$OUT/$L/${L}_${P}_matched_noise.onnx" --reference-synth "training_output/$L/eval/harvard720/synth.json" \
      --wav-dir "$OUT/$L/harvard720/$P/wavs" --output "$OUT/$L/harvard720/$P/synth.json" || fail "synth $L $P"
    done_ "synth_${L}_$P"; }
done; done

# 4. timing session on the idle machine
T=$OUT/timing
if ! have timing; then
  mkdir -p "$T"
  for i in $(seq 1 30); do                                # wait (max 30 min) for 1-min load < 1.5
    load=$(cut -d' ' -f1 /proc/loadavg)
    if ! busy && awk "BEGIN{exit !($load < 1.5)}"; then break; fi
    sleep 60
  done
  busy && fail "training process present before timing"
  state "$T/machine_state_before.txt"
  "$TPY" "$H/synthesize.py" --config configs/training/piper-no-vits2-cpn.yaml --checkpoint "$(ckpt piper-no-vits2-cpn)" \
      --manifest results/manifests/harvard720.json --wav-dir "$T/pytorch_ctrl/wavs" --output "$T/pytorch_ctrl/synth.json" \
      >>"$LOG" 2>&1 || fail "PyTorch control"
  for TH in 2 12; do
    ORDER=(); for L in "${LABELS[@]}"; do ORDER+=("$L:fp32" "$L:int8"); done
    for R in A B; do
      [ "$R" = B ] && ORDER=($(printf '%s\n' "${ORDER[@]}" | tac))
      for item in "${ORDER[@]}"; do L=${item%%:*}; P=${item##*:}
        py "$SD/time_graphs.py" --graph "$OUT/$L/${L}_$P.onnx" --threads "$TH" --round "$R" \
           --output "$T/t${TH}_${R}_${L}_$P.json" || fail "timing $L $P t$TH $R"
      done
    done
  done
  state "$T/machine_state_after.txt"
  done_ timing
fi

# 5. scoring (CPU, after timing)
for L in "${LABELS[@]}"; do for P in fp32 int8; do
  D=$OUT/$L/harvard720/$P
  have "score_${L}_$P" || { nice -n5 "$SPY" -W ignore "$H/score.py" --synth "$D/synth.json" \
      --output "$D/harvard_$(id_of "$L")_${P}_results.json" --record "$D/score_record.json" >>"$LOG" 2>&1 || fail "score $L $P"
    done_ "score_${L}_$P"; }
done; done

# 6. analysis
A=$OUT/analysis
if ! have analysis; then
  mkdir -p "$A"
  for L in "${LABELS[@]}"; do I=$(id_of "$L"); D=$OUT/$L/harvard720
    INT8=$D/int8/harvard_${I}_int8_results.json; FP32=$D/fp32/harvard_${I}_fp32_results.json
    PT=$REPO/results/$L/harvard_${I}_results.json
    py "$H/compare.py" --candidate "$INT8" --label "${I}_int8" --no-historical --extra-reference "${I}_fp32_onnx=$FP32" \
       --metrics utmosv2,wer --output "$A/${L}_int8_vs_fp32.json" || fail "compare int8 vs fp32 $L"
    py "$H/compare.py" --candidate "$FP32" --label "${I}_fp32_onnx" --no-historical --extra-reference "${I}_pytorch=$PT" \
       --metrics utmosv2,wer --output "$A/${L}_fp32_vs_pytorch.json" || fail "compare fp32 vs pytorch $L"
    py "$H/compare.py" --candidate "$INT8" --label "${I}_int8" --reference-dir "$REPO/results/historical/harvard" \
       --metrics utmosv2,wer --output "$A/${L}_int8_vs_historical.json" || fail "compare int8 vs historical $L"
  done
  py "$SD/summarize_timing.py" --timing-dir "$T" --pytorch-control "$T/pytorch_ctrl/synth.json" \
     --earlier-control "$REPO/results/piper-ir-timing-session/piper_no_vits2_cpn_ctrl_synth_record.json" \
     --output "$A/timing_summary.json" || fail "timing summary"
  done_ analysis
fi

# 7. copy the records (no graphs, no WAVs) into the repository
if ! have collect; then
  for L in "${LABELS[@]}"; do I=$(id_of "$L"); D=$OUT/$L; mkdir -p "$RESULTS/$L"
    cp "$D/export_record.json" "$D/quantize_record.json" "$RESULTS/$L/"
    for P in fp32 int8; do
      cp "$D/harvard720/$P/synth.json" "$RESULTS/$L/harvard720_${P}_synth_record.json"
      cp "$D/harvard720/$P/harvard_${I}_${P}_results.json" "$RESULTS/$L/"
      cp "$D/harvard720/$P/score_record.json" "$RESULTS/$L/${P}_score_record.json"
    done
    cp "$A/${L}_int8_vs_fp32.json" "$A/${L}_fp32_vs_pytorch.json" "$A/${L}_int8_vs_historical.json" "$RESULTS/$L/"
  done
  mkdir -p "$RESULTS/timing"
  cp "$T"/t*_*.json "$T/machine_state_before.txt" "$T/machine_state_after.txt" "$A/timing_summary.json" "$RESULTS/timing/"
  cp "$T/pytorch_ctrl/synth.json" "$RESULTS/timing/pytorch_ctrl_synth_record.json"
  done_ collect
fi
log "ALL DONE"
[ -n "${NOTIFY_USER:-}" ] && /mnt/c/Windows/System32/msg.exe "$NOTIFY_USER" "onnx-ptq round DONE; records in $RESULTS" 2>/dev/null
exit 0
