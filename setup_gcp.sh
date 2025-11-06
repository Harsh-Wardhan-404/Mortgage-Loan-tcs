#!/bin/bash

################################################################################
# GCP Instance Setup Script for Llama Training
# This script sets up a GCP VM for training Llama on regulatory compliance data
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on GCP or locally
if [[ -f /sys/class/dmi/id/product_name ]] && grep -q "Google" /sys/class/dmi/id/product_name 2>/dev/null; then
    ON_GCP=true
    print_info "Running on Google Cloud Platform"
else
    ON_GCP=false
    print_warning "Not running on GCP - this script should be run on your GCP instance"
fi

print_info "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl

print_info "Installing NVIDIA drivers and CUDA (for GPU support)..."
# Install NVIDIA drivers
sudo apt-get install -y nvidia-driver-535
sudo apt-get install -y cuda-toolkit-12-2

print_info "Setting up Python environment..."
cd ~
python3 -m venv vevn || true  # Create venv or use existing
source vevn/bin/activate

print_info "Upgrading pip..."
pip install --upgrade pip

print_info "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print_info "Installing training dependencies..."
pip install transformers datasets peft accelerate bitsandbytes sentencepiece protobuf

print_info "Verifying GPU access..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

print_info "Setup complete!"
print_info "Next steps:"
echo "  1. Upload your training data to this instance"
echo "  2. Run: python train_llama.py --train_data train_data.jsonl --val_data val_data.jsonl"
echo "  3. Monitor training progress with tensorboard"
