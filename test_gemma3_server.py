"""
Test script for the Gemma 3 OpenAI-compatible server
----------------------------------------------------

This script demonstrates how to use the OpenAI-compatible server with the Gemma 3 model.
It supports both single queries and multi-turn conversations, with options for streaming
responses and controlling generation parameters.

Features:
- Single query mode (default)
- Conversation mode for multi-turn interactions
- Streaming support for real-time responses
- Configurable generation parameters (temperature, top_p, max_tokens)
- Error handling and timeout configuration

Usage:
    # Single query
    python test_gemma3_server.py --prompt "Hello, what can you do?"
    
    # Conversation mode
    python test_gemma3_server.py --conversation
    
    # Streaming mode
    python test_gemma3_server.py --prompt "Tell me a story" --stream
"""

from openai import OpenAI
import argparse
import time
import sys
import json

def main(args):
    """
    Main function to test the Gemma 3 OpenAI-compatible server.
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Configure the client to use the local server
    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="sk-no-key-required",  # Any string will work as the API key
        timeout=60.0  # Increase timeout to 60 seconds
    )

    print(f"Connecting to server at {args.host}:{args.port}")
    
    try:
        # First check if the server is responding and get available models
        models = client.models.list()
        print(f"Available models: {[model.id for model in models.data]}")
        
        if args.conversation:
            # Run in conversation mode
            run_conversation_mode(client, args)
        else:
            # Run in single query mode
            run_single_query(client, args)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_single_query(client, args):
    """
    Run a single query to the server.
    
    Args:
        client: OpenAI client
        args: Command line arguments
    """
    print(f"Prompt: {args.prompt}")
    print("\nGenerating response (this may take a moment)...")
    start_time = time.time()
    
    # Create a chat completion
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

def run_conversation_mode(client, args):
    """
    Run an interactive conversation with the model.
    
    Args:
        client: OpenAI client
        args: Command line arguments
    """
    print("\n=== Conversation Mode ===")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'save' to save the conversation to a file")
    print("Type 'history' to show the conversation history")
    
    # Initialize conversation history
    conversation = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for special commands
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break
        elif user_input.lower() == "save":
            save_conversation(conversation)
            continue
        elif user_input.lower() == "history":
            show_conversation_history(conversation)
            continue
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Generate response
        print("\nGenerating response (this may take a moment)...")
        start_time = time.time()
        
        try:
            # Create a chat completion with the full conversation history
            response = client.chat.completions.create(
                model="gemma-3",
                messages=conversation,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                stream=args.stream
            )
            
            if args.stream:
                # Handle streaming response
                print("\nGemma: ", end="", flush=True)
                assistant_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_response += content
                        print(content, end="", flush=True)
                print()
            else:
                # Handle non-streaming response
                assistant_response = response.choices[0].message.content
                print(f"\nGemma: {assistant_response}")
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": assistant_response})
            
            end_time = time.time()
            print(f"\n[Generation time: {end_time - start_time:.2f} seconds]")
            
        except Exception as e:
            print(f"Error: {e}")

def save_conversation(conversation):
    """
    Save the conversation history to a file.
    
    Args:
        conversation: List of conversation messages
    """
    filename = f"conversation_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(conversation, f, indent=2)
    print(f"Conversation saved to {filename}")

def show_conversation_history(conversation):
    """
    Display the conversation history.
    
    Args:
        conversation: List of conversation messages
    """
    print("\n=== Conversation History ===")
    for message in conversation:
        role = message["role"].capitalize()
        content = message["content"]
        print(f"\n{role}: {content}")
    print("\n===========================")

def load_conversation(filename):
    """
    Load a conversation from a file.
    
    Args:
        filename: Path to the conversation file
        
    Returns:
        List of conversation messages
    """
    with open(filename, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the Gemma 3 OpenAI-compatible server")
    
    # Server connection parameters
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    
    # Mode selection
    parser.add_argument("--conversation", action="store_true",
                        help="Run in conversation mode (interactive)")
    parser.add_argument("--prompt", type=str, default="Hello, what can you do?",
                        help="Prompt to send to the server (used in single query mode)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter (nucleus sampling)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode for the response")
    
    # File operations
    parser.add_argument("--load", type=str,
                        help="Load conversation from file")

    args = parser.parse_args()
    main(args)