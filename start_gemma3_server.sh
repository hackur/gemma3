#!/bin/bash
#
# Start Gemma 3 OpenAI-compatible Server
# --------------------------------------
#
# This script provides a convenient way to start the Gemma 3 OpenAI-compatible server
# with configurable parameters. It handles command-line arguments, performs validation,
# and launches the server with the specified settings.
#
# Usage:
#   ./start_gemma3_server.sh [options]
#
# Options:
#   --model_path PATH    Path to the GGUF model file
#   --n_gpu_layers N     Number of GPU layers to use (-1 for all)
#   --port PORT          Port to run the server on
#   --host HOST          Host to run the server on
#   --context_length N   Context length for the model
#
# Example:
#   ./start_gemma3_server.sh --model_path /path/to/model.gguf --n_gpu_layers -1
#

# Default configuration values
MODEL_PATH="/Users/sarda/.lmstudio/models/unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf"
N_GPU_LAYERS=-1
PORT=8000
HOST="127.0.0.1"
CONTEXT_LENGTH=2048

# Display usage information
function show_usage {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --model_path PATH    Path to the GGUF model file"
  echo "  --n_gpu_layers N     Number of GPU layers to use (-1 for all)"
  echo "  --port PORT          Port to run the server on"
  echo "  --host HOST          Host to run the server on"
  echo "  --context_length N   Context length for the model"
  echo ""
  echo "Example:"
  echo "  $0 --model_path /path/to/model.gguf --n_gpu_layers -1"
  exit 1
}

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
    --context_length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --help)
      show_usage
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      ;;
  esac
done

# Validate model file exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file not found at $MODEL_PATH"
  echo "Please provide the correct path with --model_path"
  exit 1
fi

# Display server information
echo "Starting Gemma 3 OpenAI-compatible server..."
echo "Model path: $MODEL_PATH"
echo "GPU Layers: $N_GPU_LAYERS"
echo "Context Length: $CONTEXT_LENGTH"
echo "Server will be available at: http://$HOST:$PORT"
echo ""
echo "Use Ctrl+C to stop the server"
echo ""

# Start the server with the specified parameters
uv run python gemma3_server.py \
  --model_path "$MODEL_PATH" \
  --n_gpu_layers $N_GPU_LAYERS \
  --context_length $CONTEXT_LENGTH \
  --host "$HOST" \
  --port $PORT