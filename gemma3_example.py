"""
Gemma 3 Example Usage
--------------------
This script demonstrates how to use Google's Gemma 3 models via Hugging Face
or locally using GGUF format.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse
import os
from llama_cpp import Llama

def load_huggingface_model(model_name, hf_token, quantize):
    """Loads a model from Hugging Face Hub."""
    if hf_token:
        print("Using Hugging Face token for authentication")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
    )

    # Configure model loading based on quantization option
    if quantize == "4bit":
        print("Using 4-bit quantization")
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
    elif quantize == "8bit":
        print("Using 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            token=hf_token,
        )
    else:
        print("Using full precision (bfloat16)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
    return model, tokenizer

def load_gguf_model(model_path, n_gpu_layers=-1, n_ctx=2048):
    """Loads a GGUF model using llama_cpp."""
    print(f"Loading GGUF model from: {model_path}")
    model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
    return model, None  # No tokenizer needed for llama_cpp

def generate_text_huggingface(model, tokenizer, prompt, max_tokens, temperature, top_p):
    """Generates text using a Hugging Face model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    gen_time = time.time() - gen_start
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, gen_time

def generate_text_gguf(model, prompt, max_tokens, temperature, top_p, top_k, min_p, repeat_penalty):
    """Generates text using a llama_cpp model."""
    gen_start = time.time()
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        echo=False,
    )
    gen_time = time.time() - gen_start
    response = output["choices"][0]["text"]
    return response, gen_time

def main(args):
    # Get Hugging Face token from environment variable or command line
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token and args.require_token and not args.local_model:
        raise ValueError(
            "Hugging Face token is required when using a Hugging Face model, but not provided. "
            "Either set the HF_TOKEN environment variable, use the --token argument, or use --local_model."
        )

    if args.local_model:
        model, tokenizer = load_gguf_model(args.local_model, args.n_gpu_layers, args.context_length)
        generate_fn = generate_text_gguf
        # Unsloth's recommended parameters for Gemma 3
        default_temperature = 1.0
        default_top_k = 64
        default_top_p = 0.95
        default_min_p = 0.01  # or 0.0
        default_repeat_penalty = 1.0
    else:
        model, tokenizer = load_huggingface_model(args.model, hf_token, args.quantize)
        generate_fn = generate_text_huggingface
        default_temperature = 0.7
        default_top_k = None
        default_top_p = 0.9
        default_min_p = None
        default_repeat_penalty = None

    # Apply Unsloth's recommended chat template (remove <bos> for GGUF)
    if args.local_model:
      prompt = f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
      prompt = f"<bos><start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"

    print(f"\nPrompt: {prompt}")

    if args.local_model:
        response, gen_time = generate_fn(
            model,
            prompt,
            args.max_tokens,
            args.temperature if args.temperature is not None else default_temperature,
            args.top_p if args.top_p is not None else default_top_p,
            args.top_k if args.top_k is not None else default_top_k,
            args.min_p if args.min_p is not None else default_min_p,
            args.repeat_penalty if args.repeat_penalty is not None else default_repeat_penalty,
        )
    else:
        response, gen_time = generate_fn(
            model,
            tokenizer,
            prompt,
            args.max_tokens,
            args.temperature if args.temperature is not None else default_temperature,
            args.top_p if args.top_p is not None else default_top_p,
        )

    print("\nResponse:")
    print(response)
    print(f"\nGeneration time: {gen_time:.2f} seconds")
    if tokenizer:
        print(f"Generation speed: {args.max_tokens / gen_time:.2f} tokens/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 3 Example")
    parser.add_argument("--model", type=str, default="google/gemma-3-8b",
                        help="Hugging Face model name (default: google/gemma-3-8b)")
    parser.add_argument("--local_model", type=str, default=None,
                        help="Path to a local GGUF model file")
    parser.add_argument("--prompt", type=str,
                        default="Explain quantum computing in simple terms",
                        help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for sampling (default: 0.7 for HF, 1.0 for GGUF)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p sampling parameter (default: 0.9 for HF, 0.95 for GGUF)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling parameter (GGUF only, default: 64)")
    parser.add_argument("--min_p", type=float, default=None,
                        help="Min-p sampling parameter (GGUF only, default: 0.01)")
    parser.add_argument("--repeat_penalty", type=float, default=None,
                        help="Repetition penalty (GGUF only, default: 1.0)")
    parser.add_argument("--quantize", type=str, choices=["4bit", "8bit", "none"],
                        default="4bit", help="Quantization level (for Hugging Face models)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token (can also use HF_TOKEN environment variable)")
    parser.add_argument("--require_token", action="store_true",
                        help="Require Hugging Face token (will fail if not provided and using a HF model)")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="Number of GPU layers to use (-1 for all, GGUF only)")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="Context length (GGUF only)")

    args = parser.parse_args()

    if not args.local_model and not args.model:
        parser.error("Either --model or --local_model must be provided.")

    main(args)