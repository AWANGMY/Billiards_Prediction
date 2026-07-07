#!/usr/bin/env bash
set -euo pipefail
cd /work/gw17/w17001/Data/code/Billiards_Prediction
PY=/work/gw17/w17001/envs/billiards-prediction/bin/python
mkdir -p Output/reproduction/formal
CONFIGS=(
  "paper40_clean Output/reproduction/billiards_layout_paper40.pt 0.001"
  "paper40_clean Output/reproduction/billiards_layout_paper40.pt 0.0001"
  "release_code_parity Output/reproduction/billiards_layout_paper40.pt 0.001"
  "release_code_parity Output/reproduction/billiards_layout_paper40.pt 0.0001"
  "current_control Dataset/processed/billiards_layout.pt 0.001"
)
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  PROTOCOL=$1
  DATA=$2
  WD=$3
  OUT="Output/reproduction/formal/${PROTOCOL}_wd${WD}"
  mkdir -p "$OUT"
  {
    echo "protocol=$PROTOCOL"
    echo "processed_path=$DATA"
    echo "weight_decay=$WD"
    echo "start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "host=$(hostname)"
    echo "$PY -u run_blcnn_reproduction.py --processed-path $DATA --protocol $PROTOCOL --tasks all --epochs 400 --weight-decay $WD --output-dir $OUT"
  } > "$OUT/command.txt"
  echo "RUN_START $PROTOCOL wd=$WD $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "$PY" -u run_blcnn_reproduction.py \
    --processed-path "$DATA" \
    --protocol "$PROTOCOL" \
    --tasks all \
    --epochs 400 \
    --weight-decay "$WD" \
    --output-dir "$OUT" \
    > "$OUT/run.log" 2>&1
  status=$?
  echo "exit_status=$status" >> "$OUT/command.txt"
  echo "finish=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$OUT/command.txt"
  echo "RUN_DONE $PROTOCOL wd=$WD status=$status $(date '+%Y-%m-%d %H:%M:%S %Z')"
  tail -n 8 "$OUT/run.log"
done

echo "FORMAL_MATRIX_DONE $(date '+%Y-%m-%d %H:%M:%S %Z')"
