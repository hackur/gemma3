"""
Gemma 3 OpenAI-compatible Server
------------------------------
This script runs an OpenAI-compatible server using the Gemma 3 GGUF model.
It uses the llama-cpp-python CLI server directly.
"""

import argparse
import subprocess
import sys
import os

def main(args):
    # Build the command to run the llama-cpp-python server
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", args.model_path,
        "--n_gpu_layers", str(args.n_gpu_layers),
        "--n_ctx", str(args.context_length),
        "--host", args.host,
        "--port", str(args.port),
        "--chat_format", "chatml",
        "--verbose", "true"
    ]
    
    print(f"Starting Gemma 3 OpenAI-compatible server...")
    print(f"Model path: {args.model_path}")
    print(f"GPU Layers: {args.n_gpu_layers}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the server
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 3 OpenAI-compatible Server")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the GGUF model file")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="Number of GPU layers to use (-1 for all)")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="Context length")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")

    args = parser.parse_args()
    main(args)