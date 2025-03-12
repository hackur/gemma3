import argparse
import subprocess
import time
import openai
import sys
import os
import json
import requests

def start_server(model_path, n_gpu_layers, context_length, host, port):
    """Starts the llama-cpp-python server with the specified configuration."""
    cmd = [
        "uv", "run", "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--n_gpu_layers", str(n_gpu_layers),
        "--n_ctx", str(context_length),
        "--host", host,
        "--port", str(port),
        "--chat_format", "chatml",
        "--verbose", "false" # We will capture and print the output ourselves
    ]
    print(f"Starting server with command: {' '.join(cmd)}")
    # Capture stdout and stderr
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), text=True)


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

    # Print server output and error streams
    print("\nServer Output:")
    for line in process.stdout:
        print(line.strip())
    print("\nServer Error Output:")
    for line in process.stderr:
        print(line.strip())

def is_server_ready(host, port):
    """Checks if the server is ready to accept requests."""
    try:
        response = requests.get(f"http://{host}:{port}/v1/models")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def run_instruction_following_test(model_path, n_gpu_layers, context_length, instructions, max_tokens, host, port, num_tests):
    """
    Runs tests with varying instructions to assess instruction following capability.

    Args:
        model_path: Path to the GGUF model file.
        n_gpu_layers: Number of GPU layers to use.
        context_length: Context length for the server.
        instructions: A list of instruction strings to test.
        max_tokens: Maximum tokens for the response.
        host: Server host.
        port: Server port.
        num_tests: Number of test iterations for each instruction.
    """

    # Start the server
    server_process = start_server(model_path, n_gpu_layers, context_length, host, port)

    # Wait for the server to be ready
    ready = False
    attempts = 0
    while not ready and attempts < 30: # Try for 30 seconds
        ready = is_server_ready(host, port)
        if not ready:
            time.sleep(1)
            attempts += 1

    if not ready:
        print("Error: Server did not become ready after multiple attempts.")
        stop_server(server_process)
        return

    # Configure the OpenAI client
    client = openai.OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="sk-no-key-required",  # Any string will work as the API key
        timeout=120.0  # Increased timeout
    )

    results = []
    for instruction in instructions:
        print(f"Testing instruction: {instruction}")
        for i in range(num_tests):
            print(f"  Iteration: {i+1}/{num_tests}")
            start_time = time.time()

            try:
                response = client.chat.completions.create(
                    model="gemma-3",
                    messages=[
                        {"role": "user", "content": instruction}
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
                    "instruction": instruction,
                    "success": True,
                    "response_time": response_time,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "response_content": response_content,
                    "error": None
                })
                print(f"    Success! Response time: {response_time:.2f} seconds")

            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                results.append({
                    "instruction": instruction,
                    "success": False,
                    "response_time": response_time,
                    "prompt_tokens": None,
                    "response_tokens": None,
                    "response_content": None,
                    "error": str(e)
                })
                print(f"    Error: {e}")

    # Stop the server
    stop_server(server_process)
    return results

def main():
    parser = argparse.ArgumentParser(description="Instruction Following Test Suite for Gemma 3 OpenAI-Compatible Server")

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
    parser.add_argument("--instructions", type=str, default="instructions.json",
                        help="Path to a JSON file containing a list of instructions")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens for the response")
    parser.add_argument("--num_tests", type=int, default=3,
                        help="Number of test iterations for each instruction")
    parser.add_argument("--output_file", type=str, default="instruction_following_test_results.json",
                        help="Path to save the test results")

    args = parser.parse_args()

    # Load instructions from file
    with open(args.instructions, 'r') as f:
        instructions = json.load(f)

    # Run the tests
    test_results = run_instruction_following_test(
        args.model_path,
        args.n_gpu_layers,
        args.context_length,
        instructions,
        args.max_tokens,
        args.host,
        args.port,
        args.num_tests
    )

    # Save the results to a JSON file
    with open(args.output_file, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Test results saved to {args.output_file}")

if __name__ == "__main__":
    main()