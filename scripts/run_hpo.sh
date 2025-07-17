#!/bin/bash
set -e

# ---------------------------------------------------------------------
# Run Optuna HPO with dataset auto-generation from many ROOT files
# ---------------------------------------------------------------------

# --- Configurable variables -----------------------------------------
GRAPH_FILE="graphs_evan.pt"
ROOT_DIR="data"  # <-- can be folder or single file
CONFIG_FILE="configs/dataset_tau_classification.yml"
TRIALS=100
OUTFILE="best_hparams.json"

# ---------------------------------------------------------------------
echo "📂 ROOT input: $ROOT_DIR"
echo "📄 Graph file: $GRAPH_FILE"
echo "📄 Config:     $CONFIG_FILE"

# 1️⃣ Check if graphs.pt exists
if [ -f "$GRAPH_FILE" ]; then
    echo "✅ Found existing dataset → $GRAPH_FILE"
else
    echo "📦 graphs.pt not found → generating from ROOT files in '$ROOT_DIR'"

    # Check ROOT_DIR exists and has at least one .root file
    if [ ! -d "$ROOT_DIR" ] && [ ! -f "$ROOT_DIR" ]; then
        echo "❌ ROOT directory/file not found: $ROOT_DIR"
        exit 1
    fi

    if [ -d "$ROOT_DIR" ]; then
        ROOT_COUNT=$(find "$ROOT_DIR" -name "*.root" | wc -l)
        if [ "$ROOT_COUNT" -eq 0 ]; then
            echo "❌ No .root files found in directory: $ROOT_DIR"
            exit 1
        fi
    fi

    echo "🔧 Converting ROOT → PyG dataset..."
    gnn-omtf-data convert \
        --root-dir "$ROOT_DIR" \
        --config "$CONFIG_FILE" \
        --output "$GRAPH_FILE"
fi

# 2️⃣ Launch Optuna search
echo "🧠 Running Optuna (trials=$TRIALS)..."
gnn-omtf-opt hpo \
    --graphs "$GRAPH_FILE" \
    --config "$CONFIG_FILE" \
    --trials "$TRIALS" \
    --out "$OUTFILE"

echo ""
echo "✅ HPO finished. Best parameters saved to: $OUTFILE"
