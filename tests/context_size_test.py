import argparse
import subprocess
import time
import openai
import sys
import os
import json

def start_server(model_path, n_gpu_layers, context_length, host, port):
    """Starts the llama-cpp-python server with the specified configuration."""
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--n_gpu_layers", str(n_gpu_layers),
        "--n_ctx", str(context_length),
        "--host", host,
        "--port", str(port),
        "--chat_format", "chatml",
        "--verbose", "true"
    ]
    print(f"Starting server with command: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_server(process):
    """Stops the llama-cpp-python server."""
    print("Stopping server...")
    process.terminate()
    try:
        process.wait(timeout=10)  # Wait for the process to terminate
        print("Server stopped.")
    except subprocess.TimeoutExpired:
        print("Server did not stop gracefully, killing...")
        process.kill()

def run_test(model_path, n_gpu_layers, context_length, max_tokens, prompt_size, host, port, num_tests):
    """Runs a single test iteration."""
    
    # Start the server
    server_process = start_server(model_path, n_gpu_layers, context_length, host, port)
    time.sleep(5) # Wait for server to start. TODO: Implement a better way to check if server is ready.
    
    # Create a test prompt
    prompt = "a" * prompt_size
    
    # Configure the OpenAI client
    client = openai.OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="sk-no-key-required",
        timeout=60.0
    )
    
    results = []
    for i in range(num_tests):
        print(f"Running test iteration: {i+1}/{num_tests}")
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model="gemma-3",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                stream=False
            )
            end_time = time.time()
            response_time = end_time - start_time
            response_content = response.choices[0].message.content
            response_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            
            results.append({
                "success": True,
                "response_time": response_time,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_content": response_content,
                "error": None
            })
            print(f"  Success! Response time: {response_time:.2f} seconds")
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            results.append({
                "success": False,
                "response_time": response_time,
                "prompt_tokens": None,
                "response_tokens": None,
                "response_content": None,
                "error": str(e)
            })
            print(f"  Error: {e}")
    
    # Stop the server
    stop_server(server_process)
    return results

def main():
    parser = argparse.ArgumentParser(description="Context Size and Configuration Test Suite for Gemma 3 OpenAI-Compatible Server")
    
    # Server configuration
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the GGUF model file")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="Number of GPU layers to use (-1 for all)")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="Context length for the server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    
    # Test parameters
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum tokens for the response")
    parser.add_argument("--prompt_size", type=int, default=1024,
                        help="Size of the prompt (in characters)")
    parser.add_argument("--num_tests", type=int, default=5,
                        help="Number of test iterations")
    parser.add_argument("--output_file", type=str, default="test_results.json",
                        help="Path to save the test results")

    args = parser.parse_args()

    # Run the tests
    results = run_test(
        args.model_path,
        args.n_gpu_layers,
        args.context_length,
        args.max_tokens,
        args.prompt_size,
        args.host,
        args.port,
        args.num_tests
    )

    # Save the results to a JSON file
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Test results saved to {args.output_file}")

if __name__ == "__main__":
    main()