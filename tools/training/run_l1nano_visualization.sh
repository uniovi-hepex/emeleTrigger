#!/usr/bin/env bash
set -euo pipefail

ROOT_FILE_DEFAULT="/Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8.root"
OUT_PREFIX_DEFAULT="/Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/emeleTrigger/test/datasets/l1nano_evt0"

ROOT_FILE="${1:-$ROOT_FILE_DEFAULT}"
PLOT_IDX="${2:-0}"
MAX_EVENTS="${3:-5}"
OUT_PREFIX="${4:-$OUT_PREFIX_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running L1Nano visualization"
echo "  ROOT file   : $ROOT_FILE"
echo "  plot idx    : $PLOT_IDX"
echo "  max events  : $MAX_EVENTS"
echo "  out prefix  : $OUT_PREFIX"

MPLBACKEND=Agg conda run -n cmsl1t python "$SCRIPT_DIR/InputDataset.py" \
  --root_dir "$ROOT_FILE" \
  --tree_name Events \
  --max_files 1 \
  --max_events "$MAX_EVENTS" \
  --inspect_event \
  --max_print_stubs 10 \
  --max_print_edges 12 \
  --plot_stub_edge_info \
  --plot_example \
  --show_edge_attr_labels \
  --plot_idx "$PLOT_IDX" \
  --plot_output_prefix "$OUT_PREFIX"

echo "Done. Output files:"
echo "  ${OUT_PREFIX}_stub_edge.png"
echo "  ${OUT_PREFIX}_graph.png"
