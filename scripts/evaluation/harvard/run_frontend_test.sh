#!/bin/bash
# Causal frontend-mismatch test on Harvard-720 (plan: docs/plan-frontend-causal-test.md).
# Run from the training clone root:
#
#   SPY=<scorer python with banhmi_phonemize> TPY=<training venv python> PPY=<Piper env python> \
#   PIPER_CKPT=<lj-med_1000.ckpt> OUT=training_output/frontend-test RESULTS=<repository>/results/frontend-test \
#   bash <repository>/scripts/evaluation/harvard/run_frontend_test.sh
#
# Synthesizes three WAV sets (Piper with en-us phonemes as the same-session control, Piper with
# en phonemes, Piper_no_VITS2_cpn with en phonemes), scores them with harvard/score.py and
# compares them. Stage markers in $OUT/stages let a re-run resume. Nothing trains or commits.
set -uo pipefail
: "${SPY:?set SPY}" "${TPY:?set TPY}" "${PPY:?set PPY}" "${PIPER_CKPT:?set PIPER_CKPT}" "${OUT:?set OUT}" "${RESULTS:?set RESULTS}"
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)            # scripts/evaluation/harvard
REPO=$(cd "$H/../../.." && pwd)
export CUDA_VISIBLE_DEVICES=""
US=$REPO/results/manifests/harvard720.json
EN=$REPO/results/manifests/harvard720_espeak_en.json
INV=$REPO/docs/evidence/journal-evidence-package-2026-10-04/evidence/B_harvard/wav_inventory.csv
TEAM_CFG=configs/training/piper-no-vits2-cpn.yaml
TEAM_CKPT="training_output/piper-no-vits2-cpn/checkpoints/best-epoch=1403-val_loss_mel=19.7719.ckpt"
TEAM_US=$REPO/results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json
mkdir -p "$OUT/stages"
LOG=$OUT/run.log
log() { echo "[$(date '+%F %T %z')] $*" | tee -a "$LOG"; }
have() { [ -e "$OUT/stages/$1" ]; }
done_() { touch "$OUT/stages/$1"; log "STAGE $1 done"; }
fail() { log "FAILED: $1"; [ -n "${NOTIFY_USER:-}" ] && /mnt/c/Windows/System32/msg.exe "$NOTIFY_USER" "frontend test FAILED: $1" 2>/dev/null; exit 1; }
busy() { pgrep -f '^[^ ]*python[0-9.]* (-m hifimobinet\.training\.train|[^ ]*run_sequence\.py)( |$)' >/dev/null; }

busy && fail "a training process is running"
log "start; scripts from $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
[ -f "$EN" ] || fail "en manifest missing: build it with build_manifest.py and commit it"

# 1. synthesis (CPU); the Piper en-us run is the same-session control and the reproduction check
have synth_piper_enus || { "$PPY" -W ignore "$H/synthesize_piper.py" --checkpoint "$PIPER_CKPT" --manifest "$US" \
    --reference-inventory "$INV" --wav-dir "$OUT/piper_enus/wavs" --output "$OUT/piper_enus/synth.json" >>"$LOG" 2>&1 \
    || fail "Piper en-us synthesis"; done_ synth_piper_enus; }
have synth_piper_en || { "$PPY" -W ignore "$H/synthesize_piper.py" --checkpoint "$PIPER_CKPT" --manifest "$EN" \
    --wav-dir "$OUT/piper_en/wavs" --output "$OUT/piper_en/synth.json" >>"$LOG" 2>&1 \
    || fail "Piper en synthesis"; done_ synth_piper_en; }
have synth_team_en || { "$TPY" "$H/synthesize.py" --config "$TEAM_CFG" --checkpoint "$TEAM_CKPT" --manifest "$EN" \
    --wav-dir "$OUT/team_en/wavs" --output "$OUT/team_en/synth.json" >>"$LOG" 2>&1 \
    || fail "Piper_no_VITS2_cpn en synthesis"; done_ synth_team_en; }

# 2. scoring (CPU, one session)
for S in piper_en piper_enus team_en; do
  have "score_$S" || { nice -n5 "$SPY" -W ignore "$H/score.py" --synth "$OUT/$S/synth.json" \
      --output "$OUT/$S/harvard_${S}_results.json" --record "$OUT/$S/score_record.json" >>"$LOG" 2>&1 \
      || fail "score $S"; done_ "score_$S"; }
done

# 3. analysis
A=$OUT/analysis
if ! have analysis; then
  mkdir -p "$A"
  c() { "$SPY" -W ignore "$H/compare.py" --no-historical --metrics utmosv2,wer "$@" >>"$LOG" 2>&1; }
  c --candidate "$OUT/piper_en/harvard_piper_en_results.json" --label piper_en \
    --extra-reference "piper_enus=$OUT/piper_enus/harvard_piper_enus_results.json" --output "$A/piper_en_vs_piper_enus.json" \
    || fail "compare Piper en vs en-us"
  c --candidate "$OUT/team_en/harvard_team_en_results.json" --label team_en \
    --extra-reference "team_enus=$TEAM_US" --output "$A/team_en_vs_team_enus.json" || fail "compare team en vs en-us"
  c --candidate "$TEAM_US" --label team_enus \
    --extra-reference "piper_en=$OUT/piper_en/harvard_piper_en_results.json" --output "$A/team_enus_vs_piper_en.json" \
    || fail "compare team vs Piper en"
  for S in piper_enus piper_en; do
    (cd "$REPO" && "$SPY" -W ignore "$H/dialect_check.py" --candidate "$S=$OLDPWD/$OUT/$S/harvard_${S}_results.json" \
       --control "piper_no_vits2_cpn=$TEAM_US" --output "$OLDPWD/$A/dialect_check_$S.json") >>"$LOG" 2>&1 \
       || fail "dialect check $S"
  done
  done_ analysis
fi

# 4. copy the records (no WAVs) into the repository
if ! have collect; then
  mkdir -p "$RESULTS"
  for S in piper_en piper_enus team_en; do
    cp "$OUT/$S/synth.json" "$RESULTS/${S}_synth_record.json"
    cp "$OUT/$S/harvard_${S}_results.json" "$RESULTS/"
    cp "$OUT/$S/score_record.json" "$RESULTS/${S}_score_record.json"
  done
  cp "$A"/*.json "$RESULTS/"
  done_ collect
fi
log "ALL DONE"
[ -n "${NOTIFY_USER:-}" ] && /mnt/c/Windows/System32/msg.exe "$NOTIFY_USER" "frontend test DONE; records in $RESULTS" 2>/dev/null
exit 0
