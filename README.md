# Gemma 3 Setup and Usage

This repository provides everything you need to work with Google's Gemma 3 models using Hugging Face and uv for environment management. It also supports loading local GGUF models using llama.cpp.

## Important Note

Gemma 3 models require a specialized version of the transformers library. This repository is configured to use the correct version:

```bash
git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

Some Gemma 3 models may require authentication with a Hugging Face API token.

## Contents

- `requirements.txt` - All required dependencies, including `llama-cpp-python`
- `setup_gemma3_env.sh` - Shell script to set up the environment using uv
- `gemma3_example.py` - Example script demonstrating how to use Gemma 3 models (both Hugging Face and GGUF)
- `gemma3_setup.py` - Detailed information about dependencies and setup options
- `gemma3_report_2025-03-12_084449/` - Detailed documentation and project reports

## Setup Instructions

### Prerequisites

- Python 3.9+ installed
- Git (for cloning this repository)
- Internet connection (for downloading models and dependencies)
- (Optional) Hugging Face API token (for accessing gated models)
- (Optional) C++ compiler (for building `llama-cpp-python` from source, if needed)

### Quick Setup

The easiest way to set up your environment is to use the provided shell script:

```bash
# Make the script executable
chmod +x setup_gemma3_env.sh

# Run the setup script
./setup_gemma3_env.sh
```

This script will:

1. Install uv if not already installed
2. Create a virtual environment
3. Install all required dependencies (including the specialized transformers version and `llama-cpp-python`)
4. Verify the installation
5. Check for the presence of the `HF_TOKEN` environment variable

### Manual Setup

If you prefer to set up the environment manually:

```bash
# Install uv (if not already installed)
curl -sSf https://install.ultraviolet.dev | sh

# Create a virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Hugging Face API Token

Some Gemma 3 models may require authentication with a Hugging Face API token. You can obtain a token from your Hugging Face account settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

You can provide the token in two ways:

1. **Environment Variable (Recommended)**: Set the `HF_TOKEN` environment variable:

   ```bash
   export HF_TOKEN=your_hugging_face_token
   ```

2. **Command-Line Argument**: Use the `--token` argument when running `gemma3_example.py`:

   ```bash
   python gemma3_example.py --token your_hugging_face_token
   ```

The setup script will check for the `HF_TOKEN` environment variable and provide a warning if it's not set.

## Using Gemma 3 Models

### Running the Example Script

After setting up the environment, you can run the example script:

```bash
# Activate the environment (if not already activated)
source .venv/bin/activate
```

### Using Hugging Face Models

To use a Hugging Face model:

```bash
# Run with default settings (Gemma 3 8B, 4-bit quantization)
python gemma3_example.py

# Use a different model
python gemma3_example.py --model google/gemma-3-27b

# Use the instruction-tuned 1B model
python gemma3_example.py --model google/gemma-3-1b-it

# Use a custom prompt
python gemma3_example.py --prompt "Write a short poem about AI"

# Adjust generation parameters
python gemma3_example.py --max_tokens 1024 --temperature 0.9 --top_p 0.95

# Change quantization level
python gemma3_example.py --quantize 8bit  # Options: 4bit, 8bit, none

# Provide Hugging Face token (if not using HF_TOKEN environment variable)
python gemma3_example.py --token your_hugging_face_token
```

### Using Local GGUF Models

To use a local GGUF model:

```bash
# Run with a local GGUF model
python gemma3_example.py --local_model /path/to/your/model.gguf

# Specify the number of GPU layers (use -1 for all available layers)
python gemma3_example.py --local_model /path/to/your/model.gguf --n_gpu_layers 40

# Adjust context length
python gemma3_example.py --local_model /path/to/your/model.gguf --context_length 4096

# Use a custom prompt
python gemma3_example.py --local_model /path/to/your/model.gguf --prompt "Write a short story"
```

**Note**: When using a local GGUF model, the `--model`, `--quantize`, and `--token` arguments are ignored.

## Available Models

### Hugging Face Models

Gemma 3 is available in different sizes on Hugging Face:

- `google/gemma-3-8b` - 8 billion parameter model (recommended for most users)
- `google/gemma-3-27b` - 27 billion parameter model (higher quality, requires more resources)
- `google/gemma-3-1b-it` - 1 billion parameter instruction-tuned model (fastest, good for simple tasks)

### GGUF Models

You can find GGUF models for Gemma 3 on Hugging Face, often in repositories from users like TheBloke. You'll need to download these models separately.

## Hardware Requirements

### Hugging Face Models

The hardware requirements depend on the model size and quantization level:

#### 4-bit Quantization (default)

- **8B model**: 6GB+ VRAM
- **27B model**: 20GB+ VRAM
- **1B-IT model**: 2GB+ VRAM

#### 8-bit Quantization

- **8B model**: 12GB+ VRAM
- **27B model**: 40GB+ VRAM
- **1B-IT model**: 4GB+ VRAM

#### Full Precision (no quantization)

- **8B model**: 24GB+ VRAM
- **27B model**: 80GB+ VRAM
- **1B-IT model**: 8GB+ VRAM

### GGUF Models

The hardware requirements for GGUF models depend on the specific quantization used (e.g., Q4_K_M, Q5_K_M, etc.). Refer to the documentation for the specific GGUF model you're using. Generally, GGUF models with smaller quantization levels require less VRAM.

## Using in Your Own Code

### Hugging Face Model Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Load tokenizer and model
model_name = "google/gemma-3-8b"  # or "google/gemma-3-27b" or "google/gemma-3-1b-it"

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Use 4-bit quantization for efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    token=hf_token,
)

# Generate text
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### GGUF Model Example

```python
from llama_cpp import Llama

# Load the GGUF model
model_path = "/path/to/your/model.gguf"
model = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048) # Use -1 for all GPU layers

# Generate text
prompt = "Explain quantum computing in simple terms"
output = model(
    prompt,
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    echo=False,
)
response = output["choices"][0]["text"]
print(response)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**

   - Try using a smaller model (e.g., 1B-IT instead of 8B, or a GGUF model with smaller quantization)
   - Use 4-bit quantization instead of 8-bit or full precision (for Hugging Face models)
   - Reduce batch size or sequence length
   - Reduce the context length (`--context_length`) for GGUF models
   - Increase the number of GPU layers (`--n_gpu_layers`) for GGUF models if you have more VRAM available

2. **Slow Generation**

   - Use a smaller model
   - Reduce the number of generated tokens
   - Ensure you're using GPU acceleration if available
   - Use a GGUF model with appropriate quantization for your hardware

3. **Installation Problems**

   - Make sure you're using the specialized transformers version
   - Check that all dependencies are installed correctly
   - Verify that your Python version is 3.9 or higher
   - If installing `llama-cpp-python` from source, ensure you have a C++ compiler installed

4. **Authentication Errors**

   - Ensure you have a valid Hugging Face API token
   - Set the `HF_TOKEN` environment variable correctly
   - Use the `--token` argument when running the example script

5. **Model Not Found Errors**
   - Double-check the model name (for Hugging Face models) or path (for GGUF models)
   - Make sure you've downloaded the GGUF model file

### Getting Help

For more detailed information, refer to the documentation in the `gemma3_report_2025-03-12_084449/` directory:

- `completion_report.md` - Comprehensive project overview
- `project_summary.md` - Quick reference guide
- `todo.md` - Future enhancements and known limitations

## License

This project is provided as-is under the MIT License. The Gemma 3 models themselves are subject to Google's model license, which you should review before using the models.

## Acknowledgments

- Google for creating the Gemma 3 models
- Hugging Face for providing model hosting and the transformers library
- The uv team for creating an excellent environment management tool
- The llama.cpp team for creating a powerful inference engine for GGUF models
