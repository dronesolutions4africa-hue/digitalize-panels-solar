#!/bin/bash
export PATH=/home/solar/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SITE=/home/solar/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$SITE/cublas/lib:$SITE/cuda_cupti/lib:$SITE/cuda_nvrtc/lib:$SITE/cuda_runtime/lib:$SITE/cudnn/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/nccl/lib:$SITE/nvjitlink/lib
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$SCRIPT_DIR"

python3 train.py \
  --model fast_scnn_2 \
  --ortho Data/Orthomosaic_Patisen.tif \
  --shp Data/Panneaux_Patisen.shp \
  --tile_size 512 \
  --stride 256 \
  --batch_size 8 \
  --epochs 50 \
  --panel_oversample 4 \
  --output_dir trained_models/patisen_gpu
