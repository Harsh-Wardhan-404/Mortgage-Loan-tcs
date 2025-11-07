# Quick Start - Train Llama on Regulatory Data

## 🚀 Three-Step Process

### Step 1: Request GPU Quota
```bash
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Search: "NVIDIA T4" or "GPU"
# Request: 1 GPU quota increase
# Wait: 24-48 hours for approval
```

### Step 2: Deploy to GCP
```bash
# Edit the script first
nano deploy_to_gcp.sh
# Set PROJECT_ID="your-project-id"

# Run deployment
./deploy_to_gcp.sh
```

### Step 3: Train
```bash
# SSH into instance
gcloud compute ssh llama-training --zone=us-central1-a

# Run training
source vevn/bin/activate
python train_llama.py --train_data train_data.jsonl --val_data val_data.jsonl
```

## ⚡ What Happens

```
Your Data → Tokenize → Llama 3.1 8B (frozen) + LoRA (trainable) → Trained Model
```