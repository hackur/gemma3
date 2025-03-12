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
#   --test_script SCRIPT   Specify the test script to run (e.g., context_size_test.py)
#   --extra_args ARGS      Extra arguments to pass to the test script (quoted string)
#
# Example:
#   ./run_tests.sh --model_path /path/to/model.gguf --n_gpu_layers -1 --test_script context_size_test.py --extra_args "--context_length 4096 --max_tokens 512"
#

# Default configuration values
MODEL_PATH=""
N_GPU_LAYERS=-1
CONTEXT_LENGTH=2048
MAX_TOKENS=128
PROMPT_SIZE=1024  # Default, will be overridden by specific tests if needed.
NUM_TESTS=5
OUTPUT_FILE="test_results.json"
HOST="127.0.0.1"
PORT=8000
TEST_SCRIPT=""  # No default test script
EXTRA_ARGS=""

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
  echo "  --test_script SCRIPT   Specify the test script to run (e.g., context_size_test.py)"
  echo '  --extra_args ARGS      Extra arguments to pass to the test script (quoted string, e.g. "--arg1 val1 --arg2 val2")'
  echo "  --help                 Display this help message"
  echo ""
  echo "Example:"
  echo '  $0 --model_path /path/to/model.gguf --n_gpu_layers -1 --test_script context_size_test.py --extra_args "--context_length 4096 --max_tokens 512"'
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
    --test_script)
      TEST_SCRIPT="$2"
      shift 2
      ;;
    --extra_args)
      EXTRA_ARGS="$2"
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

# Run the tests, specifying the output file in the tests directory and the test script
if [ -z "$TEST_SCRIPT" ]; then
  echo "Error: --test_script must be specified."
  show_usage
fi

uv run python "tests/$TEST_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --n_gpu_layers "$N_GPU_LAYERS" \
  --context_length "$CONTEXT_LENGTH" \
  --max_tokens "$MAX_TOKENS" \
  --prompt_size "$PROMPT_SIZE" \
  --num_tests "$NUM_TESTS" \
  --output_file "tests/$OUTPUT_FILE" \
  --host "$HOST" \
  --port "$PORT" \
  $EXTRA_ARGS

echo ""
echo "Tests completed. Results saved to tests/$OUTPUT_FILE"