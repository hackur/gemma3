# Gemma 3 OpenAI-Compatible Server

This repository provides a complete solution for running Google's Gemma 3 models locally with an OpenAI-compatible API server. It leverages `llama-cpp-python` for efficient inference and `uv` for environment management.

## Features

-   **OpenAI API Compatibility:** Use the server with any OpenAI client library or tools designed for the OpenAI API.
-   **Gemma 3 Support:** Optimized for Gemma 3 models in GGUF format.
-   **GPU Acceleration:**  Configurable GPU layer offloading for optimal performance.
-   **Conversation Mode:** Supports multi-turn conversations with history.
-   **Streaming Responses:**  Get real-time token generation for responsive applications.
-   **Easy Setup:**  Simple installation and configuration with `uv` and a convenient start script.
-   **Test Suite:** Includes a test suite for evaluating performance and context size limits.

## Contents

-   `requirements.txt`:  All required dependencies.
-   `gemma3_server.py`:  The main server implementation using `llama-cpp-python`.
-   `test_gemma3_server.py`:  A client script to test the server's functionality.
-   `start_gemma3_server.sh`:  A shell script to easily start the server.
-   `GEMMA3_SERVER_README.md`: Detailed documentation for the server.
-   `tests/`:  Directory containing the test suite.
    -   `tests/context_size_test.py`:  Script for testing context size and configuration parameters.
    -   `tests/run_tests.sh`:  Script to run the test suite.
    -   `tests/TODO.md`: TODO list for test suite development.
-   `setup_gemma3_env.sh`: (Legacy) Shell script to set up the environment using uv.
-   `gemma3_example.py`: (Legacy) Example script demonstrating how to use Gemma 3 models (both Hugging Face and GGUF).
-   `gemma3_setup.py`: (Legacy) Detailed information about dependencies and setup options.

## Setup

### Prerequisites

-   Python 3.8+
-   A compatible GGUF Gemma 3 model file (e.g., from Unsloth or TheBloke on Hugging Face).
-   (Recommended) A GPU with sufficient VRAM for your chosen model and quantization.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**

    ```bash
    uv pip install -r requirements.txt
    ```
    This installs:
    - `llama-cpp-python[server]`: The core library, including server components.
    - `uvicorn`: An ASGI server to run the FastAPI application.
    - `openai`: The OpenAI Python client library (for testing).
    - `fastapi`: The web framework used by the server.
    - `starlette`: ASGI toolkit.
    - Other necessary dependencies.

## Running the Server

The recommended way to start the server is with the provided `start_gemma3_server.sh` script:

```bash
./start_gemma3_server.sh --model_path /path/to/your/gemma-3-model.gguf --n_gpu_layers -1
```

Replace `/path/to/your/gemma-3-model.gguf` with the actual path to your downloaded GGUF model file.  `--n_gpu_layers -1` offloads all layers to the GPU (recommended if you have enough VRAM).

**Available Options:**

-   `--model_path`:  **(Required)** Path to the GGUF model file.
-   `--n_gpu_layers`:  Number of layers to offload to the GPU.  Use -1 to offload all layers. (Default: -1)
-   `--context_length`:  The maximum context size (in tokens) the model can handle. (Default: 2048)
-   `--host`:  The host address to bind the server to. (Default: 127.0.0.1)
-   `--port`:  The port to run the server on. (Default: 8000)

The server will be accessible at `http://127.0.0.1:8000` (or the host/port you specified).

## Testing the Server

### Basic Test Script

A simple test script (`test_gemma3_server.py`) is included to verify the server is working:

```bash
uv run python test_gemma3_server.py --prompt "Hello, what can you do?"
```

This sends a single prompt to the server and prints the response.

**Options:**

- `--prompt`: The prompt text.
- `--conversation`:  Run in interactive conversation mode.
- `--stream`: Enable streaming responses.
- `--temperature`:  Adjust the randomness of the generated text (higher values are more random).
- `--top_p`:  Use nucleus sampling (controls the diversity of the generated text).
- `--max_tokens`:  Limit the maximum number of tokens in the response.

### Interactive Conversation Mode

```bash
uv run python test_gemma3_server.py --conversation
```

This starts an interactive chat session.  You can type messages, and the server will respond, maintaining conversation history.  Use `exit` or `quit` to end the session, `save` to save the conversation, and `history` to view the conversation history.

### Test Suite

A more comprehensive test suite is available in the `tests` directory.  This suite allows you to test different configurations and measure performance.

```bash
./tests/run_tests.sh --model_path /path/to/your/gemma-3-model.gguf --n_gpu_layers -1
```

**Options:**

The `tests/run_tests.sh` script uses the same options as `start_gemma3_server.sh`, plus:

-   `--max_tokens`:  Maximum tokens for the generated responses.
-   `--prompt_size`:  The size of the prompt (in characters) to use for testing.
-   `--num_tests`:  The number of test iterations to run.
-   `--output_file`:  The file to save the test results to (default: `tests/test_results.json`).

The test suite starts the server, sends requests with varying prompt sizes, measures response times, and checks for errors.  The results are saved to a JSON file.

## Using the API

The server provides an OpenAI-compatible API.  You can use it with any OpenAI client library (like the `openai` Python package) by setting the `base_url` to your server's address.

**Example (Python):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # Replace with your server's address
    api_key="sk-no-key-required",  # Any string works as the API key
)

# Single turn
response = client.chat.completions.create(
    model="gemma-3",  # The model name doesn't matter; the server uses the loaded model
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Multi-turn conversation
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]
response = client.chat.completions.create(
    model="gemma-3",
    messages=messages,
    max_tokens=50,
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "What is its population?"})
response = client.chat.completions.create(
    model="gemma-3",
    messages=messages,
    max_tokens=50,
)
print(response.choices[0].message.content)

```

**cURL Example:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
```

For detailed API documentation, refer to the `GEMMA3_SERVER_README.md` file.

## Using with Roo Code

1.  Start the server using `start_gemma3_server.sh`.
2.  Configure Roo Code to use the following settings:
    *   **API Base URL:** `http://127.0.0.1:8000/v1` (or the address where your server is running)
    *   **API Key:**  Any string (e.g., "sk-no-key-required")

## Troubleshooting

See `GEMMA3_SERVER_README.md` for troubleshooting tips and common issues.
