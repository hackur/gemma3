# Context Size and Configuration Test Suite - TODO

This document outlines the tasks for creating a test suite to evaluate the performance and limitations of the Gemma 3 OpenAI-compatible server.

## Tasks

1.  **Create `tests` directory:** (DONE)
2.  **Create `TODO.md`:** (DONE)
3.  **Create `context_size_test.py` script:**
    *   Implement argument parsing for:
        *   `--model_path`: Path to the GGUF model file
        *   `--n_gpu_layers`: Number of GPU layers
        *   `--context_length`: Context length for the server
        *   `--max_tokens`: Maximum tokens for the response
        *   `--prompt_size`: Size of the prompt (in tokens or characters)
        *   `--num_tests`: Number of test iterations
        *   `--output_file`: Path to save the test results
        *   `--host`: Server host
        *   `--port`: Server port
    *   Implement server startup and shutdown:
        *   Use `subprocess` to start the server with the specified configuration.
        *   Ensure the server is shut down properly after the tests.
    *   Implement request generation:
        *   Create a function to generate prompts of varying sizes.
        *   Create requests with different `max_tokens` values.
    *   Implement response handling:
        *   Send requests to the server using the `openai` client.
        *   Record response times.
        *   Check for errors (e.g., exceeding context window).
        *   Store responses and metadata.
    *   Implement report generation:
        *   Create a function to generate a report summarizing the test results.
        *   Include information on:
            *   Configuration parameters (context size, max tokens, etc.)
            *   Prompt sizes
            *   Response times
            *   Success/failure status
            *   Error messages (if any)
        *   Save the report to a file (e.g., CSV, Markdown, JSON).
4.  **Update `GEMMA3_SERVER_README.md`:** (DONE)
    *   Add a section describing the test suite.
    *   Explain how to run the tests.
    *   Provide examples of usage.
5. **Create shell script to run tests** (DONE)
6. **Create `tests/prompt_length_test.py`:**
    * Test server's handling of increasingly long prompts.
    * Measure response time and success/failure.
    * Check for truncation or errors near the context limit.
7. **Create `tests/repetition_test.py`:**
    * Test model's tendency to repeat itself.
    * Vary `repeat_penalty` and `temperature`.
    * Analyze output for repetition patterns.
8. **Create `tests/instruction_following_test.py`:**
    * Test model's ability to follow specific instructions.
    * Use prompts with constraints (e.g., length, format, content).
    * Evaluate the accuracy and completeness of the responses.
9. **Update `tests/run_tests.sh`:**
    * Add options to run specific test scripts.
    * Provide a way to run all tests.