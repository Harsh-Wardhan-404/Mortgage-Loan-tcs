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

print_info "Running environment check..."
if [[ -f /sys/class/dmi/id/product_name ]] && grep -q "Google" /sys/class/dmi/id/product_name 2>/dev/null; then
    print_info "Running on Google Cloud Platform"
else
    print_warning "Not running on GCP - this script is intended for the deployed VM"
fi

print_info "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git curl rsync

print_info "Setting up Python environment..."
cd ~
if [[ ! -d venv ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

print_info "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt tensorboard

print_info "Checking for training data overrides via GCS URIs..."
if [[ -n "${GCS_TRAIN_DATA_URI:-}" ]]; then
    print_info "Downloading training data from ${GCS_TRAIN_DATA_URI}"
    gsutil cp "${GCS_TRAIN_DATA_URI}" ~/train_data.jsonl
fi
if [[ -n "${GCS_VAL_DATA_URI:-}" ]]; then
    print_info "Downloading validation data from ${GCS_VAL_DATA_URI}"
    gsutil cp "${GCS_VAL_DATA_URI}" ~/val_data.jsonl
fi

print_info "Creating helper scripts..."
cat <<'EOF' > start_tensorboard.sh
#!/bin/bash
source venv/bin/activate
LOGDIR=${1:-~/llama_regulatory_model/runs}
PORT=${TENSORBOARD_PORT:-6006}
HOST=${TENSORBOARD_HOST:-0.0.0.0}
mkdir -p "$LOGDIR"
nohup tensorboard --logdir="$LOGDIR" --port="$PORT" --host="$HOST" > ~/tensorboard.log 2>&1 &
echo "TensorBoard started on port $PORT (log: ~/tensorboard.log)"
EOF
chmod +x start_tensorboard.sh

cat <<'EOF' > upload_model_to_gcs.sh
#!/bin/bash
if [[ -z "$1" ]]; then
    echo "Usage: ./upload_model_to_gcs.sh gs://bucket/path"
    exit 1
fi
OUTPUT_DIR=${MODEL_OUTPUT_DIR:-~/llama_regulatory_model}
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Model output directory $OUTPUT_DIR not found."
    exit 2
fi
DEST=$1
echo "Uploading $OUTPUT_DIR to $DEST"
gsutil -m cp -r "$OUTPUT_DIR" "$DEST"
EOF
chmod +x upload_model_to_gcs.sh

print_info "Setup complete!"
print_info "Next steps:"
echo "  1. Activate env: source venv/bin/activate"
echo "  2. Run training: python train_llama.py --train_data train_data.jsonl --val_data val_data.jsonl --epochs 3 --batch_size 4 --fp16"
echo "  3. (Optional) Start TensorBoard: ./start_tensorboard.sh"
echo "  4. (Optional) Upload model to GCS: ./upload_model_to_gcs.sh gs://your-bucket/path"
