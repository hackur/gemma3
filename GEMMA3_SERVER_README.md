# Gemma 3 OpenAI-Compatible Server

This project provides a fully OpenAI-compatible API server for the Gemma 3 model using GGUF format. It allows you to run Gemma 3 locally with the same API interface as OpenAI, making it compatible with any tools or libraries that work with the OpenAI API.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Testing the Server](#testing-the-server)
- [Using the API](#using-the-api)
  - [Python Example](#python-example)
  - [cURL Example](#curl-example)
  - [Conversation Example](#conversation-example)
- [Using with Roo Code](#using-with-roo-code)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Features

- **Full OpenAI API Compatibility**: Works with any OpenAI client library
- **GPU Acceleration**: Configurable GPU layer offloading for optimal performance
- **Conversation Support**: Maintains conversation history for multi-turn interactions
- **Streaming Responses**: Real-time token streaming for responsive applications
- **Configurable Parameters**: Control temperature, top_p, and other generation settings
- **Easy Setup**: Simple installation and configuration

## Prerequisites

- Python 3.8+
- GGUF model file for Gemma 3 (e.g., `gemma-3-27b-it-Q4_K_M.gguf`)
- GPU with sufficient VRAM (recommended for optimal performance)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone or download this repository

2. Install the required dependencies:

```bash
uv pip install -r requirements.txt
```

This will install:
- `llama-cpp-python[server]`: The core library for running the model with OpenAI API compatibility
- `uvicorn`: ASGI server for running the API
- `openai`: Python client for testing the API
- Other dependencies required for running the model

## Running the Server

### Using the Convenience Script

The easiest way to start the server is using the provided shell script:

```bash
./start_gemma3_server.sh --model_path /path/to/your/gemma-3-27b-it-Q4_K_M.gguf --n_gpu_layers -1
```

### Manual Start

Alternatively, you can start the server directly with Python:

```bash
uv run python gemma3_server.py --model_path /path/to/your/gemma-3-27b-it-Q4_K_M.gguf --n_gpu_layers -1
```

### Parameters

- `--model_path`: Path to your GGUF model file (required)
- `--n_gpu_layers`: Number of layers to offload to GPU (-1 for all)
- `--context_length`: Context window size (default: 2048)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 8000)

### Example with Specific Settings

```bash
./start_gemma3_server.sh --model_path /Users/sarda/.lmstudio/models/unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf --n_gpu_layers -1 --context_length 4096
```

## Testing the Server

A test script is provided to verify that the server is working correctly:

```bash
uv run python test_gemma3_server.py --prompt "Hello, what can you do?"
```

### Test Script Options

- `--prompt`: The prompt to send to the server (for single query mode)
- `--conversation`: Run in interactive conversation mode
- `--stream`: Use streaming mode for responses
- `--temperature`: Temperature for sampling (default: 1.0)
- `--top_p`: Top-p sampling parameter (default: 0.95)
- `--max_tokens`: Maximum tokens to generate (default: 512)

### Interactive Conversation Mode

To test multi-turn conversations:

```bash
uv run python test_gemma3_server.py --conversation
```

This will start an interactive session where you can chat with the model. Special commands:
- Type `exit` or `quit` to end the conversation
- Type `save` to save the conversation to a file
- Type `history` to show the conversation history

## Using the API

The server provides a fully OpenAI-compatible API. You can use it with any OpenAI client library by setting the base URL to your local server.

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
    top_p=0.95,
    max_tokens=512
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
    "top_p": 0.95,
    "max_tokens": 512
  }'
```

### Conversation Example

For multi-turn conversations, include the full conversation history in the messages array:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required"
)

# First message
response = client.chat.completions.create(
    model="gemma-3",
    messages=[
        {"role": "user", "content": "Hello, who are you?"}
    ],
    temperature=1.0
)

assistant_response = response.choices[0].message.content
print(f"Assistant: {assistant_response}")

# Second message (including conversation history)
response = client.chat.completions.create(
    model="gemma-3",
    messages=[
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": "Can you help me with Python programming?"}
    ],
    temperature=1.0
)

print(f"Assistant: {response.choices[0].message.content}")
```

## Using with Roo Code

To use this server with Roo Code:

1. Start the server as described above
2. Configure Roo Code to use your local API endpoint:
   - Set the API Base URL to `http://localhost:8000/v1`
   - Use any string as the API key (e.g., `sk-no-key-required`)

## API Reference

The server implements the following OpenAI API endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create a chat completion
- `POST /v1/completions` - Create a completion

### Chat Completions Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| model | string | Model name (any string works) |
| messages | array | Array of message objects with role and content |
| temperature | number | Sampling temperature (default: 1.0) |
| top_p | number | Nucleus sampling parameter (default: 0.95) |
| max_tokens | integer | Maximum tokens to generate (default: 512) |
| stream | boolean | Whether to stream the response |
| stop | string/array | Stop sequences to end generation |

## Performance Considerations

- **GPU Layers**: Setting `--n_gpu_layers -1` offloads all layers to GPU for maximum performance
- **Context Length**: Larger context lengths require more memory but allow for longer conversations
- **Quantization**: The GGUF model is already quantized for efficient memory usage

## Troubleshooting

### Server Won't Start

- Verify the model path is correct
- Check that you have sufficient GPU memory
- Ensure all dependencies are installed correctly

### Slow Response Times

- Try reducing the number of GPU layers if you have limited VRAM
- Reduce the context length
- Use a more aggressively quantized model (e.g., Q4_K_M instead of Q5_K_M)

### API Connection Issues

- Verify the server is running and accessible at the specified host and port
- Check that your client is using the correct base URL
- Ensure you're providing an API key (any string will work)