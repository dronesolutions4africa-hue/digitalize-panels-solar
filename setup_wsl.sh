#!/bin/bash
# setup_wsl.sh — Install Python + TensorFlow-GPU + geospatial stack in WSL2 Ubuntu
# Run once after Ubuntu 22.04 is installed:
#   bash /mnt/c/Users/user/Downloads/solar-panels-detection-master/solar-panels-detection-master/setup_wsl.sh

set -e  # stop on first error

echo "============================================================"
echo " Solar Panel Detection — WSL2 environment setup"
echo "============================================================"

# 1. System packages
echo ""
echo "[1/5] Updating apt and installing system deps..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    build-essential libssl-dev libffi-dev \
    libgl1 libglib2.0-0

# 2. Create virtual environment
VENV="$HOME/solar_env"
echo ""
echo "[2/5] Creating virtual environment at $VENV ..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel --quiet

# 3. TensorFlow with CUDA (bundled CUDA 12.x libs — no separate CUDA install needed)
echo ""
echo "[3/5] Installing TensorFlow[and-cuda] (this may take a few minutes)..."
pip install "tensorflow[and-cuda]==2.18.0" --quiet

# 4. Geospatial + training packages
echo ""
echo "[4/5] Installing geospatial + ML packages..."
pip install \
    "rasterio>=1.3" \
    "geopandas>=0.14" \
    "shapely>=2.0" \
    "opencv-python-headless>=4.8" \
    "tqdm>=4.66" \
    "matplotlib>=3.8" \
    "numpy>=1.26" \
    "h5py>=3.10" \
    --quiet

# 5. Verify GPU
echo ""
echo "[5/5] Verifying GPU detection..."
python3 - <<'PYEOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"  SUCCESS: {len(gpus)} GPU(s) detected")
    for g in gpus:
        print(f"    {g}")
else:
    print("  WARNING: No GPU detected — check NVIDIA driver in Windows (>= 525.xx required)")
PYEOF

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " To activate the environment in future sessions:"
echo "   source ~/solar_env/bin/activate"
echo ""
echo " To start training:"
echo "   cd /mnt/c/Users/user/Downloads/solar-panels-detection-master/solar-panels-detection-master"
echo "   source ~/solar_env/bin/activate"
echo "   python train.py"
echo "============================================================"
