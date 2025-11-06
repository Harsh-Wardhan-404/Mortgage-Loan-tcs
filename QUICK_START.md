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
Your Data → Tokenize → Llama 3 8B (frozen) + LoRA (trainable) → Trained Model
```

- **Input**: Your regulatory Q&A pairs (train_data.jsonl)
- **Model**: Llama 3 8B Instruct with 4-bit quantization
- **Training**: Only 8M parameters (LoRA adapters)
- **Output**: Specialized regulatory compliance model
- **Time**: 6-8 hours
- **Cost**: ~$5

## 📊 Memory Comparison

| Approach | Memory | Trainable Params | Cost/6hr |
|----------|--------|-----------------|----------|
| Full Fine-tuning | 70GB | 7B | $50+ |
| LoRA (ours) | 4GB | 8M | $5 |

## 🎯 What You Get

1. **Trained model** in `llama_regulatory_model/`
2. **Checkpoints** saved every 500 steps
3. **TensorBoard logs** for monitoring
4. **Production-ready** regulatory Q&A model

## 💻 Using Your Model

```python
# Load and use
python inference_example.py \
    --model_path ./llama_regulatory_model \
    --question "What are housing finance eligibility criteria?"
```

## 📚 Full Documentation

- **GCP_TRAINING_GUIDE.md** - Complete step-by-step guide
- **TRAINING_SUMMARY.md** - Technical details and architecture
- **README.md** - Updated with training info

## ⚠️ Before Starting

1. ✅ GCP account with billing enabled
2. ✅ PROJECT_ID set in deploy script
3. ✅ GPU quota approved
4. ✅ Training data ready (train_data.jsonl, val_data.jsonl)

## 💡 Tips

- **Monitor**: `tensorboard --logdir llama_regulatory_model/runs`
- **Check GPU**: `nvidia-smi`
- **View Logs**: `tail -f llama_regulatory_model/logs.txt`
- **Save costs**: Stop instance when not training
