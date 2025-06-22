#!/bin/bash
# Script to fix NumPy and PyTorch compatibility issues

echo "Fixing NumPy and PyTorch compatibility issues..."

# Create a virtual environment for clean installation
echo "Creating virtual environment..."
python -m venv venv_whisperx
source venv_whisperx/bin/activate

# Uninstall packages that might have compatibility issues
echo "Uninstalling potentially problematic packages..."
pip uninstall -y numpy transformers torch torchaudio torchvision whisperx numba

# Install NumPy 1.x first
echo "Installing NumPy < 2.0.0..."
pip install "numpy<2.0.0"

# Install PyTorch with CUDA support (adjust based on your CUDA version)
echo "Installing PyTorch with CUDA support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other ML dependencies
echo "Installing transformers..."
pip install transformers

# Reinstall the requirements
echo "Reinstalling requirements..."
pip install -r requirements/prod.txt

echo "Installation complete. NumPy and PyTorch compatibility should be fixed."
echo "Activate the virtual environment with: source venv_whisperx/bin/activate"