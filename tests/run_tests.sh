#!/bin/bash
#
# Run Context Size and Configuration Tests
# ----------------------------------------
#
# This script provides a convenient way to run the context size and configuration
# tests for the Gemma 3 OpenAI-compatible server.
#
# Usage:
#   ./run_tests.sh [options]
#
# Options:
#   --model_path PATH      Path to the GGUF model file (required)
#   --n_gpu_layers N       Number of GPU layers to use (-1 for all)
#   --context_length N     Context length for the server
#   --max_tokens N         Maximum tokens for the response
#   --prompt_size N        Size of the prompt in characters
#   --num_tests N          Number of test iterations
#   --output_file PATH     Path to save the test results
#   --host HOST            Server host
#   --port PORT            Server port
#   --help                 Display this help message
#
# Example:
#   ./run_tests.sh --model_path /path/to/model.gguf --n_gpu_layers -1 --context_length 4096 --max_tokens 512 --num_tests 10
#

# Default configuration values
MODEL_PATH=""
N_GPU_LAYERS=-1
CONTEXT_LENGTH=2048
MAX_TOKENS=128
PROMPT_SIZE=1024
NUM_TESTS=5
OUTPUT_FILE="test_results.json"
HOST="127.0.0.1"
PORT=8000

# Display usage information
function show_usage {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --model_path PATH      Path to the GGUF model file (required)"
  echo "  --n_gpu_layers N       Number of GPU layers to use (-1 for all)"
  echo "  --context_length N     Context length for the server"
  echo "  --max_tokens N         Maximum tokens for the response"
  echo "  --prompt_size N        Size of the prompt in characters"
  echo "  --num_tests N          Number of test iterations"
  echo "  --output_file PATH     Path to save the test results"
  echo "  --host HOST            Server host"
  echo "  --port PORT            Server port"
  echo "  --help                 Display this help message"
  echo ""
  echo "Example:"
  echo "  $0 --model_path /path/to/model.gguf --n_gpu_layers -1 --context_length 4096 --max_tokens 512 --num_tests 10"
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
    --context_length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --prompt_size)
      PROMPT_SIZE="$2"
      shift 2
      ;;
    --num_tests)
      NUM_TESTS="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
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
if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model_path is required"
  show_usage
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file not found at $MODEL_PATH"
  echo "Please provide the correct path with --model_path"
  exit 1
fi

# Run the tests
echo "Running context size and configuration tests..."
echo "Model path: $MODEL_PATH"
echo "GPU Layers: $N_GPU_LAYERS"
echo "Context Length: $CONTEXT_LENGTH"
echo "Max Tokens: $MAX_TOKENS"
echo "Prompt Size: $PROMPT_SIZE"
echo "Number of Tests: $NUM_TESTS"
echo "Output File: $OUTPUT_FILE"
echo ""

# Run the tests
python tests/context_size_test.py \
  --model_path "$MODEL_PATH" \
  --n_gpu_layers "$N_GPU_LAYERS" \
  --context_length "$CONTEXT_LENGTH" \
  --max_tokens "$MAX_TOKENS" \
  --prompt_size "$PROMPT_SIZE" \
  --num_tests "$NUM_TESTS" \
  --output_file "$OUTPUT_FILE" \
  --host "$HOST" \
  --port "$PORT"

echo ""
echo "Tests completed. Results saved to $OUTPUT_FILE"