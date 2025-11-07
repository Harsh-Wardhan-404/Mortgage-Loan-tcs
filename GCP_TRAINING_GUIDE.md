# Llama Training on GCP - Complete Guide

This guide walks you through training Llama on regulatory compliance data using Google Cloud Platform.

## Overview

We're using **LoRA (Low-Rank Adaptation)** for efficient fine-tuning, which allows us to train large models with minimal GPU memory. This approach:

- Reduces memory usage by 95%
- Trains only ~1% of model parameters
- Maintains model quality
- Makes training feasible on single GPU instances

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Input: Regulatory QnA Data (train_data.jsonl)        │
└──────────────────┬────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Llama Model (7B parameters)                           │
│  ├─ Frozen Base Layers (7B params)                     │
│  └─ LoRA Adapters (8-16M trainable params)             │
└──────────────────┬────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Output: Fine-tuned Model for Regulatory Compliance     │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **GCP Account** with billing enabled
2. **Project** created with GPU quota requested
3. **gcloud CLI** installed locally
4. **Hugging Face token** for model access

## Step 1: Request GPU Quota

```bash
# Request GPU quota (may take 24-48 hours to approve)
gcloud compute project-info add-metadata \
    --metadata quota-metric='compute.googleapis.com/nvidia_tesla_t4_1' \
    --metadata quota-value='1'
```

Or use the GCP Console:
1. Go to "IAM & Admin" → "Quotas"
2. Search for "NVIDIA T4"
3. Request quota increase

## Step 2: Set Up Local Environment

```bash
# Clone or navigate to your project
cd /path/to/your/project

# Make scripts executable
chmod +x deploy_to_gcp.sh setup_gcp.sh

# Edit deploy_to_gcp.sh to set your PROJECT_ID
nano deploy_to_gcp.sh
```

## Step 3: Deploy to GCP

Option A: Use the deployment script

```bash
# Run the deployment script
./deploy_to_gcp.sh
```

Option B: Manual deployment

```bash
# 1. Create instance
gcloud compute instances create llama-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE

# 2. SSH into instance
gcloud compute ssh llama-training --zone=us-central1-a

# 3. On the instance, run setup
./setup_gcp.sh
```

## Step 4: Transfer Your Data

```bash
# From your local machine, copy training data
gcloud compute scp \
    train_data.jsonl val_data.jsonl train_llama.py requirements.txt \
    llama-training:~/ --zone=us-central1-a
```

## Step 5: Start Training

SSH into your instance and run training:

```bash
gcloud compute ssh llama-training --zone=us-central1-a

# Activate environment
source vevn/bin/activate

# Run training
python train_llama.py \
    --train_data train_data.jsonl \
    --val_data val_data.jsonl \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --output_dir ./llama_regulatory_model \
    --fp16
```

## Training Parameters Explained

### Model Configuration

- **Model**: Llama 3.1 8B Instruct (meta-llama/Meta-Llama-3.1-8B-Instruct)
  - Pre-trained on conversational data
  - Good foundation for Q&A tasks
  - 8 billion parameters

- **Quantization**: 4-bit (BitsAndBytes)
  - Reduces memory from 14GB → 4GB
  - Minimal performance loss
  - Enables training on consumer GPUs

### LoRA Configuration

- **Rank (r)**: 8
  - Lower rank = fewer parameters
  - Balance between quality and efficiency
  
- **Alpha (α)**: 16
  - Scaling factor for LoRA weights
  - Rule of thumb: α = 2×r

- **Dropout**: 0.05
  - Prevents overfitting
  - Regularization technique

- **Target Modules**: Query, Key, Value, Output projections
  - These are the attention layers
  - Most impactful for fine-tuning

### Training Configuration

- **Batch Size**: 4
  - Small for memory efficiency
  - GPU memory limits this

- **Gradient Accumulation**: 4
  - Effective batch size = 4 × 4 = 16
  - Simulates larger batches

- **Learning Rate**: 2e-4
  - Conservative for fine-tuning
  - Prevents catastrophic forgetting

- **Max Length**: 2048 tokens
  - Your instruction + input + output
  - Balance context vs. memory

- **Optimizer**: Paged AdamW 8-bit
  - Memory-efficient optimizer
  - Faster convergence

## Monitoring Training

### TensorBoard

```bash
# On your GCP instance
tensorboard --logdir ./llama_regulatory_model/runs

# From local machine (port forwarding)
gcloud compute ssh llama-training --zone=us-central1-a \
    --ssh-flag="-L 6006:localhost:6006"

# Then access http://localhost:6006
```

### Watch GPU Usage

```bash
# Monitor GPU
watch -n 1 nvidia-smi
```

### Check Logs

```bash
# View training logs
tail -f llama_regulatory_model/logs.txt
```

## Cost Estimation

**Instance**: n1-standard-8 with NVIDIA T4
- $0.35/hour for instance
- $0.35/hour for T4 GPU
- **Total**: ~$0.70/hour

**Training Time**: ~6-8 hours for 3 epochs
- **Total Cost**: ~$4.50 - $5.60

To stop the instance and save costs:
```bash
gcloud compute instances stop llama-training --zone=us-central1-a
```

## Understanding the Training Process

### What Happens During Training?

1. **Forward Pass**: Model processes your regulatory Q&A
2. **Loss Calculation**: Compare predicted vs. actual output
3. **Backward Pass**: Calculate gradients for LoRA parameters
4. **Update**: Adjust only the LoRA adapters (8M params)
5. **Repeat**: For each batch, across all epochs

### Why LoRA Works

Traditional fine-tuning:
- Updates all 7B parameters
- Requires 70GB+ GPU memory
- Very expensive

LoRA fine-tuning:
- Adds low-rank matrices (A × B = ΔW)
- Updates only adapters (~8M params)
- Requires 4GB GPU memory
- Much cheaper

### Data Flow

```
Instruction → Tokenizer → Model (with LoRA) → Generated Output
                      ↓
    Compare with Ground Truth → Compute Loss → Backprop
```

## Troubleshooting

### Out of Memory (OOM) Errors

```python
# Reduce batch size
--batch_size 2

# Enable gradient checkpointing
# Add to TrainingArguments:
gradient_checkpointing=True
```

### Training Too Slow

```python
# Increase batch size with gradient accumulation
--batch_size 8 --gradient_accumulation_steps 2
```

### Model Not Converging

```python
# Lower learning rate
--learning_rate 1e-4

# Train for more epochs
--epochs 5
```

## Using the Fine-tuned Model

After training, you'll have a model checkpoint in `llama_regulatory_model/`.

### Load and Use Locally

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, "./llama_regulatory_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Generate response
prompt = "What are the eligibility criteria for housing finance loans?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Merge LoRA Weights (Optional)

To create a standalone model without PEFT:

```python
# Merge adapters into base model
base_model = model.get_base_model()
base_model.save_pretrained("./merged_model")
```

## Next Steps

1. **Evaluate** your model on a test set
2. **Compare** with base Llama
3. **Iterate** on training parameters
4. **Deploy** for production use

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## Support

For issues or questions:
1. Check GPU drivers: `nvidia-smi`
2. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check logs in output directory
4. Review TensorBoard metrics
