#!/bin/bash
# Setup script for Gemma 3 environment using uv

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -sSf https://install.ultraviolet.dev | sh
    
    # Add uv to PATH for this session if not automatically added
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment
echo "Creating virtual environment with uv..."
uv venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; import transformers; print(f'PyTorch version: {torch.__version__}'); print(f'Transformers version: {transformers.__version__}')"

echo ""
echo "Environment setup complete!"
echo "To activate this environment in the future, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run the example Gemma 3 code, execute:"
echo "python gemma3_example.py"