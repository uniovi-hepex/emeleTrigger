#!/bin/bash
set -e  # exit on error

# -----------------------------------------------------------------------------
# Create and activate a Python virtual environment for the GNN-OMTF project
# -----------------------------------------------------------------------------

ENV_DIR=".venv"
PYTHON_VERSION="3.10"

echo "🔧 Creating virtual environment in '$ENV_DIR' with Python $PYTHON_VERSION..."

# Check if venv already exists
if [ -d "$ENV_DIR" ]; then
    echo "⚠️  Virtual environment '$ENV_DIR' already exists. Skipping creation."
else
    # Create virtual environment
    python$PYTHON_VERSION -m venv "$ENV_DIR"
fi

# Activate the environment
echo "📡 Activating environment..."
source "$ENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install project dependencies (editable + extras if present)
echo "📦 Installing project dependencies..."
pip install -e ".[dev,notebook]"  # adjust extras as needed

# Final message
echo ""
echo "✅ Virtual environment is ready and activated."
echo "To activate it later, run:"
echo "    source $ENV_DIR/bin/activate"
echo ""
echo "You can now run:"
echo "    gnn-omtf-opt hpo --graphs graphs.pt --trials 40 --out best_hparams.json"
