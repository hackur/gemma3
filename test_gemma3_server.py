"""
Test script for the Gemma 3 OpenAI-compatible server
"""

from openai import OpenAI
import argparse
import time
import sys

def main(args):
    # Configure the client to use the local server
    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="sk-no-key-required",  # Any string will work as the API key
        timeout=60.0  # Increase timeout to 60 seconds
    )

    print(f"Sending request to {args.host}:{args.port}")
    print(f"Prompt: {args.prompt}")
    
    try:
        # First check if the server is responding
        models = client.models.list()
        print(f"Available models: {[model.id for model in models.data]}")
        
        # Create a chat completion
        print("\nGenerating response (this may take a moment)...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gemma-3",  # Model name doesn't matter, server will use the loaded model
            messages=[
                {"role": "user", "content": args.prompt}
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stream=args.stream
        )
        
        if args.stream:
            # Handle streaming response
            print("\nResponse:")
            for chunk in response:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            end_time = time.time()
        else:
            # Handle non-streaming response
            end_time = time.time()
            print("\nResponse:")
            print(response.choices[0].message.content)
        
        print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Gemma 3 OpenAI-compatible server")
    parser.add_argument("--prompt", type=str, default="Think deeply and return the meaning of life.",
                        help="Prompt to send to the server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode for the response")

    args = parser.parse_args()
    main(args)