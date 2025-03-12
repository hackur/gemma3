# Gemma 3 OpenAI-Compatible Server

This project provides an OpenAI-compatible API server for the Gemma 3 model using GGUF format.

## Prerequisites

- Python 3.8+
- GGUF model file for Gemma 3 (e.g., `gemma-3-27b-it-Q4_K_M.gguf`)
- Dependencies listed in `requirements.txt`

## Installation

Install the required dependencies:

```bash
uv pip install -r requirements.txt
```

## Running the Server

Start the server with the following command:

```bash
uv run python gemma3_server.py --model_path /path/to/your/gemma-3-27b-it-Q4_K_M.gguf --n_gpu_layers -1
```

Parameters:

- `--model_path`: Path to your GGUF model file (required)
- `--n_gpu_layers`: Number of layers to offload to GPU (-1 for all)
- `--context_length`: Context window size (default: 2048)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 8000)

Example with your specific settings:

```bash
uv run python gemma3_server.py --model_path /Users/sarda/.lmstudio/models/unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf --n_gpu_layers -1
```

## Using the API

The server provides an OpenAI-compatible API. You can use it with any OpenAI client library by setting the base URL to your local server.

### Python Example

```python
from openai import OpenAI

# Configure the client to use your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required"  # Any string will work as the API key
)

# Create a chat completion
response = client.chat.completions.create(
    model="gemma-3",  # Model name doesn't matter, server will use the loaded model
    messages=[
        {"role": "user", "content": "Think deeply and return the meaning of life."}
    ],
    temperature=1.0,
    top_p=0.95
)

print(response.choices[0].message.content)
```

### cURL Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-no-key-required" \
  -d '{
    "model": "gemma-3",
    "messages": [
      {
        "role": "user",
        "content": "Think deeply and return the meaning of life."
      }
    ],
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

## Using with Roo Code

To use this server with Roo Code:

1. Start the server as described above
2. Configure Roo Code to use your local API endpoint:
   - Set the API base URL to `http://localhost:8000/v1`
   - Use any string as the API key (e.g., `sk-no-key-required`)

## Notes

- The server uses the same recommended parameters as in the example script: temperature=1.0, top_k=64, top_p=0.95, min_p=0.01, repeat_penalty=1.0
- The chat format is set to "chatml" which is compatible with Gemma 3's format
- The server exposes standard OpenAI API endpoints including `/v1/chat/completions` and `/v1/completions`