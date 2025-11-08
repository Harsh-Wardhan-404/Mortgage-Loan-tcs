#!/bin/bash

################################################################################
# Deploy to GCP Script
# This script helps you create a GCP instance and deploy training code
################################################################################

# Configuration
PROJECT_ID="tcs-model-training"  # Set your GCP project ID
ZONE="us-east1-c"
INSTANCE_NAME="llama-training"
MACHINE_TYPE="n1-standard-16"  # 16 vCPUs, 60GB RAM (safer for preprocessing)
BOOT_DISK_SIZE="200GB"
# Use a specific DLVM family known to exist (PyTorch 2.1 + CUDA 11.8, Ubuntu 20.04)
# To discover other families: gcloud compute images list --project=deeplearning-platform-release | grep pytorch
IMAGE_NAME="pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20251105"
IMAGE_PROJECT="deeplearning-platform-release"
ACCELERATOR_TYPE="nvidia-tesla-t4"
ACCELERATOR_COUNT="1"
TENSORBOARD_RULE="allow-tensorboard-6006"

print_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    print_error "Please set PROJECT_ID in this script"
    exit 1
fi

set -e

print_info "Setting up GCP project..."
gcloud config set project $PROJECT_ID

print_info "Ensuring TensorBoard firewall rule exists (tcp:6006)..."
if ! gcloud compute firewall-rules describe $TENSORBOARD_RULE >/dev/null 2>&1; then
    gcloud compute firewall-rules create $TENSORBOARD_RULE \
        --allow tcp:6006 \
        --source-ranges 0.0.0.0/0 \
        --description "Allow TensorBoard access" \
        --quiet
fi

print_info "Creating GPU instance for training..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --image=$IMAGE_NAME \
    --image-project=$IMAGE_PROJECT \
    --accelerator type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.read_write

print_info "Waiting for instance to be ready..."
sleep 30

print_info "Copying files to instance..."
gcloud compute scp \
    --zone=$ZONE \
    train_llama.py train_data.jsonl val_data.jsonl requirements.txt setup_gcp.sh \
    ${INSTANCE_NAME}:~/

print_info "SSH'ing into instance to run setup..."
gcloud compute ssh \
    --zone=$ZONE \
    $INSTANCE_NAME \
    --command="chmod +x setup_gcp.sh && ./setup_gcp.sh"

print_info "Deployment complete!"
print_info "To SSH into the instance:"
echo "  gcloud compute ssh --zone=$ZONE $INSTANCE_NAME"
print_info "To run training:"
echo "  python train_llama.py --train_data train_data.jsonl --val_data val_data.jsonl"
print_info "To start TensorBoard (already installed):"
echo "  source venv/bin/activate && ./start_tensorboard.sh"
