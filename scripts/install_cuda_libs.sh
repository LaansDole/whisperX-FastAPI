#!/bin/bash

# This script helps install the necessary CUDA libraries for whisperX
# It attempts to identify your system and provide appropriate installation commands

set -e  # Exit immediately if a command fails

echo "=== CUDA Libraries Installation Helper ==="
echo "This script will help you install the required CUDA libraries for whisperX"
echo

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
        echo "Detected distribution: $DISTRO $VERSION"
        return 0
    else
        echo "Could not detect Linux distribution"
        return 1
    fi
}

# Function to install on Ubuntu/Debian
install_ubuntu() {
    echo "Installing cuDNN libraries for Ubuntu/Debian..."
    echo "You may be prompted for sudo password"
    
    # Add CUDA repository if needed
    if [ ! -f /etc/apt/sources.list.d/cuda.list ]; then
        echo "Adding CUDA repository..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        rm cuda-keyring_1.0-1_all.deb
        sudo apt-get update
    fi
    
    # Install the required libraries
    sudo apt-get install -y libcudnn8 libcudnn8-dev
    
    # Verify installation
    if ldconfig -p | grep -q libcudnn; then
        echo "[OK] cuDNN installation successful"
    else
        echo "[FAIL] cuDNN installation may have failed"
    fi
}

# Function to install on CentOS/RHEL
install_centos() {
    echo "Installing cuDNN libraries for CentOS/RHEL..."
    
    # Check if EPEL is installed
    if ! rpm -qa | grep -q epel-release; then
        sudo yum install -y epel-release
    fi
    
    # Add CUDA repository if needed
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    
    # Install the required libraries
    sudo yum install -y libcudnn8 libcudnn8-devel
    
    # Verify installation
    if ldconfig -p | grep -q libcudnn; then
        echo "[OK] cuDNN installation successful"
    else
        echo "[FAIL] cuDNN installation may have failed"
    fi
}

# Function for manual installation instructions
manual_instructions() {
    echo "=== Manual Installation Instructions ==="
    echo "To manually install the required CUDA libraries:"
    echo
    echo "1. Visit the NVIDIA cuDNN download page:"
    echo "   https://developer.nvidia.com/cudnn"
    echo
    echo "2. Create an NVIDIA Developer account if you don't have one"
    echo
    echo "3. Download the appropriate cuDNN package for your CUDA version"
    echo
    echo "4. Install the package following NVIDIA's instructions"
    echo
    echo "5. Make sure the libraries are in the system's library path"
    echo "   You may need to add the cuDNN library directory to LD_LIBRARY_PATH:"
    echo "   export LD_LIBRARY_PATH=/path/to/cudnn/lib:\$LD_LIBRARY_PATH"
    echo
}

# Function to check if Docker is running
check_docker() {
    if [ -S /var/run/docker.sock ]; then
        echo "It appears Docker is running on this system."
        echo
        echo "If you're running whisperX in a Docker container, make sure to:"
        echo "1. Use the '--gpus all' flag when starting the container"
        echo "2. Use an image with CUDA and cuDNN pre-installed"
        echo "   Example: docker run --gpus all -p 8000:8000 whisperx-service"
        echo
        echo "=== Docker-specific Installation ==="
        echo "If you're building a custom Docker image, add this to your Dockerfile:"
        echo
        echo "FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
        echo "# ... your other Dockerfile commands"
        echo
    fi
}

# Main installation flow
main() {
    # Check if running in Docker
    check_docker
    
    # Try to detect distribution and install
    if detect_distro; then
        case $DISTRO in
            ubuntu|debian|pop|mint|elementary)
                install_ubuntu
                ;;
            centos|rhel|fedora|rocky|almalinux)
                install_centos
                ;;
            *)
                echo "Automatic installation not supported for $DISTRO"
                manual_instructions
                ;;
        esac
    else
        manual_instructions
    fi
    
    echo
    echo "=== Alternative: Use CPU Mode ==="
    echo "If you prefer not to install CUDA libraries, you can run whisperX in CPU mode:"
    echo "1. Set environment variable: export DEVICE=cpu"
    echo "2. Set environment variable: export COMPUTE_TYPE=int8"
    echo "3. Restart the application"
    echo
    echo "Note: CPU mode will be significantly slower than GPU acceleration."
}

# Run the main function
main
