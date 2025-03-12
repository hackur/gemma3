#!/bin/bash
# Script to start the Gemma 3 OpenAI-compatible server

# Default model path - change this to your model path if needed
MODEL_PATH="/Users/sarda/.lmstudio/models/unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf"
N_GPU_LAYERS=-1
PORT=8000
HOST="127.0.0.1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --n_gpu_layers)
      N_GPU_LAYERS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file not found at $MODEL_PATH"
  echo "Please provide the correct path with --model_path"
  exit 1
fi

echo "Starting Gemma 3 OpenAI-compatible server..."
echo "Model path: $MODEL_PATH"
echo "GPU Layers: $N_GPU_LAYERS"
echo "Server will be available at: http://$HOST:$PORT"

# Start the server
uv run python gemma3_server.py \
  --model_path "$MODEL_PATH" \
  --n_gpu_layers $N_GPU_LAYERS \
  --host "$HOST" \
  --port $PORT