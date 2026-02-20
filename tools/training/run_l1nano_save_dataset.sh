#!/usr/bin/env bash
set -euo pipefail

ROOT_FILE_DEFAULT="/Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8_L1NanoWithGenPropagated_20260212.root"
OUT_FILE_DEFAULT="/Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/emeleTrigger/test/datasets/l1nano_graphs_withGenPropagated.pt"

ROOT_FILE="${1:-$ROOT_FILE_DEFAULT}"
MAX_EVENTS="${2:--1}"
MAX_FILES="${3:-1}"
OUT_FILE="${4:-$OUT_FILE_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "$OUT_FILE")"

echo "Saving L1Nano dataset"
echo "  ROOT file : $ROOT_FILE"
echo "  max events: $MAX_EVENTS"
echo "  max files : $MAX_FILES"
echo "  output    : $OUT_FILE"

conda run -n cmsl1t python "$SCRIPT_DIR/InputDataset.py" \
  --root_dir "$ROOT_FILE" \
  --tree_name Events \
  --max_files "$MAX_FILES" \
  --max_events "$MAX_EVENTS" \
  --save_path "$OUT_FILE" --debug

echo "Done. Dataset saved at: $OUT_FILE"
