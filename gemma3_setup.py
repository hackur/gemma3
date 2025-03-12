"""
Gemma 3 Setup and Dependencies
-----------------------------
This file contains all the necessary dependencies and setup instructions
for working with Google's Gemma 3 models via Hugging Face, using uv for
environment management.
"""

# Dependencies for Gemma 3
dependencies = [
    "torch>=2.2.0",           # PyTorch for model operations
    "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3",  # Special Gemma 3 version of Transformers
    "accelerate>=0.27.0",     # For optimized inference
    "huggingface_hub>=0.20.0", # For downloading models from HF
    "sentencepiece>=0.1.99",  # For tokenization
    "protobuf>=4.25.0",       # Required for some model operations
    "bitsandbytes>=0.41.0",   # For quantization support
    "safetensors>=0.4.0",     # For secure model loading
    "einops>=0.7.0",          # For tensor operations
    "tqdm>=4.66.0",           # For progress bars
    "numpy>=1.24.0",          # For numerical operations
]

# Example usage of Gemma 3 model
example_code = """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
model_name = "google/gemma-3-8b"  # or "google/gemma-3-27b" or "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
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
"""

# Environment setup instructions using uv
uv_setup_instructions = """
# Setting up environment with uv

## Install uv (if not already installed)
curl -sSf https://install.ultraviolet.dev | sh

## Create a new virtual environment
uv venv .venv

## Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\\Scripts\\activate

## Install dependencies from requirements.txt
# The requirements.txt file contains the special version of transformers needed for Gemma 3:
# git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

uv pip install -r requirements.txt
"""

# Quantization options for running on consumer hardware
quantization_info = """
# Quantization Options for Gemma 3

For running Gemma 3 models on consumer hardware, you can use quantization:

## 4-bit Quantization
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-8b",
    device_map="auto",
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
)

## 8-bit Quantization
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-8b",
    device_map="auto",
    load_in_8bit=True,
)
"""

if __name__ == "__main__":
    print("Gemma 3 Setup and Dependencies")
    print("------------------------------")
    print("\nDependencies:")
    for dep in dependencies:
        print(f"- {dep}")
    
    print("\nTo set up your environment with uv, follow these instructions:")
    print(uv_setup_instructions)
    
    print("\nExample code for using Gemma 3:")
    print(example_code)
    
    print("\nQuantization options for consumer hardware:")
    print(quantization_info)
    
    # Create requirements.txt file
    with open("requirements.txt", "w") as f:
        for dep in dependencies:
            f.write(f"{dep}\n")
    print("\nCreated requirements.txt with all dependencies.")
    
    print("\nTo install dependencies with uv, run:")
    print("uv venv .venv")
    print("source .venv/bin/activate  # On macOS/Linux")
    print("uv pip install -r requirements.txt")