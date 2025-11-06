#!/bin/bash

################################################################################
# Deploy to GCP Script
# This script helps you create a GCP instance and deploy training code
################################################################################

# Configuration
PROJECT_ID=""  # Set your GCP project ID
ZONE="us-central1-a"
INSTANCE_NAME="llama-training"
MACHINE_TYPE="n1-standard-8"  # 8 vCPUs, 30GB RAM
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
ACCELERATOR_TYPE="nvidia-tesla-t4"
ACCELERATOR_COUNT="1"

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

print_info "Creating GPU instance for training..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --accelerator type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform

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
