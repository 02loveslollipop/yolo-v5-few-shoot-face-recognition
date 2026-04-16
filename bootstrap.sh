#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
YOLOV5_DIR="$ROOT_DIR/external/yolov5"

echo "Bootstrapping delivery bundle in: $ROOT_DIR"

if [ ! -d "$YOLOV5_DIR/.git" ]; then
  echo "Cloning YOLOv5 into $YOLOV5_DIR"
  mkdir -p "$ROOT_DIR/external"
  git clone --depth 1 https://github.com/ultralytics/yolov5.git "$YOLOV5_DIR"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "Installing web server requirements"
python -m pip install -r "$ROOT_DIR/web_app_onnx/requirements.txt"

echo "Installing YOLOv5 runtime requirements"
python -m pip install -r "$YOLOV5_DIR/requirements.txt"

if [ ! -f "$ROOT_DIR/model/best.pt" ]; then
  echo "Warning: $ROOT_DIR/model/best.pt is missing."
fi

if [ ! -f "$ROOT_DIR/web_app_onnx/model/best.onnx" ]; then
  echo "Warning: $ROOT_DIR/web_app_onnx/model/best.onnx is missing."
fi

echo "Bootstrap complete."
echo "Activate the environment with: source \"$VENV_DIR/bin/activate\""
