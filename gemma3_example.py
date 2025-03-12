"""
Gemma 3 Example Usage
--------------------
This script demonstrates how to use Google's Gemma 3 models via Hugging Face.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse

def main(args):
    print(f"Loading Gemma 3 model: {args.model}")
    start_time = time.time()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Configure model loading based on quantization option
    if args.quantize == "4bit":
        print("Using 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
        )
    elif args.quantize == "8bit":
        print("Using 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            load_in_8bit=True,
        )
    else:
        print("Using full precision (bfloat16)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Generate text
    prompt = args.prompt
    print(f"\nPrompt: {prompt}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    gen_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_time = time.time() - gen_start
    
    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\nResponse:")
    print(response)
    print(f"\nGeneration time: {gen_time:.2f} seconds")
    print(f"Generation speed: {args.max_tokens / gen_time:.2f} tokens/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 3 Example")
    parser.add_argument("--model", type=str, default="google/gemma-3-8b", 
                        help="Model name (default: google/gemma-3-8b)")
    parser.add_argument("--prompt", type=str, 
                        default="Explain quantum computing in simple terms",
                        help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--quantize", type=str, choices=["4bit", "8bit", "none"],
                        default="4bit", help="Quantization level")
    
    args = parser.parse_args()
    main(args)