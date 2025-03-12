"""
Gemma 3 OpenAI-compatible Server
------------------------------
This script runs an OpenAI-compatible server using the Gemma 3 GGUF model.
It uses the llama-cpp-python CLI server directly to provide a fully compatible
OpenAI API that can be used with any OpenAI client library.

Features:
- Full OpenAI API compatibility
- Support for chat completions with conversation history
- GPU acceleration with configurable layer offloading
- Configurable context length and other parameters
- Optimized for Gemma 3 models with chatml format

Usage:
    python gemma3_server.py --model_path /path/to/model.gguf --n_gpu_layers -1

API Endpoints:
- GET /v1/models - List available models
- POST /v1/chat/completions - Create a chat completion
- POST /v1/completions - Create a completion
"""

import argparse
import subprocess
import sys
import os

def main(args):
    """
    Main function to start the OpenAI-compatible server.
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Build the command to run the llama-cpp-python server
    # The server provides a fully OpenAI-compatible API
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        # Model configuration
        "--model", args.model_path,
        "--n_gpu_layers", str(args.n_gpu_layers),
        "--n_ctx", str(args.context_length),
        # Server configuration
        "--host", args.host,
        "--port", str(args.port),
        # Gemma 3 specific configuration
        "--chat_format", "chatml",  # Use the chatml format for Gemma 3
        # Additional options
        "--verbose", "true"
    ]
    
    # Print server information
    print(f"Starting Gemma 3 OpenAI-compatible server...")
    print(f"Model path: {args.model_path}")
    print(f"GPU Layers: {args.n_gpu_layers}")
    print(f"Context Length: {args.context_length}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the server
    try:
        # This will block until the server is stopped
        subprocess.run(cmd)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nServer stopped by user")
    except Exception as e:
        # Handle any other errors
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gemma 3 OpenAI-compatible Server")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the GGUF model file")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="Number of GPU layers to use (-1 for all)")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="Context length for the model (max tokens in context)")
    
    # Server parameters
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")

    # Parse arguments and start the server
    args = parser.parse_args()
    main(args)