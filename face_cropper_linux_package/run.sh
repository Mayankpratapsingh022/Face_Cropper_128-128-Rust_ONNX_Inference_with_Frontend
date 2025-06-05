#!/bin/bash
set -e

# -----------------------------
# Step 1: Fix symlinks for runtime .so files
# -----------------------------
cd "$(dirname "$0")"

cd runtimeLib

# Remove existing (possibly broken) symlinks
rm -f libonnxruntime.so libonnxruntime.so.1

# Create correct symlinks
ln -sf libonnxruntime.so.1.16.0 libonnxruntime.so.1
ln -sf libonnxruntime.so.1       libonnxruntime.so

cd ..

# -----------------------------
# Step 2: Set runtime library path
# -----------------------------
export LD_LIBRARY_PATH=$(pwd)/runtimeLib:$LD_LIBRARY_PATH

# -----------------------------
# Step 3: Run the binary with arguments
# -----------------------------
./face_cropper \
  --input-dir ./input_images \
  --output-dir ./faces_cropped \
  --model ./model.onnx \
  --batch-size 10 \
  --conf-threshold 0.5 \
  --iou-threshold 0.5 \
  --brightness-jitter 30 \
  --no-recursive
