# Llama Training on GCP - Quick Summary

## What We Built

A complete system to fine-tune Llama models on regulatory compliance data using:
- **LoRA** for efficient training (1% of parameters)
- **4-bit quantization** for memory efficiency
- **GCP GPU instances** for cost-effective training

## Files Created

### Core Training
- `train_llama.py` - Main training script with LoRA support
- `inference_example.py` - Example of using trained model
- `requirements.txt` - Updated with training dependencies

### GCP Deployment
- `setup_gcp.sh` - Setup script for GCP instance
- `deploy_to_gcp.sh` - Automated deployment script
- `GCP_TRAINING_GUIDE.md` - Comprehensive training guide

## Quick Start

### 1. Request GPU Quota
```bash
# Via GCP Console: IAM & Admin → Quotas → Search "NVIDIA T4" → Request increase
```

### 2. Deploy to GCP
```bash
# Edit deploy_to_gcp.sh and set PROJECT_ID
nano deploy_to_gcp.sh

# Run deployment
./deploy_to_gcp.sh
```

### 3. Train Your Model
```bash
# SSH into instance
gcloud compute ssh llama-training --zone=us-central1-a

# Run training
cd ~
source vevn/bin/activate
python train_llama.py \
    --train_data train_data.jsonl \
    --val_data val_data.jsonl \
    --epochs 3 \
    --batch_size 4 \
    --fp16
```

## Training Time & Cost

- **Time**: 6-8 hours for 3 epochs
- **Cost**: ~$5 for complete training
- **Instance**: n1-standard-8 + NVIDIA T4
- **Memory Usage**: ~4GB GPU memory (with 4-bit)

## How It Works

```
1. Load Llama 3.1 8B with 4-bit quantization (~16GB → 4GB)
2. Add LoRA adapters to attention layers (~8M parameters)
3. Freeze base model, train only LoRA weights
4. Save checkpoints during training
5. Evaluate on validation set
```

## Key Technologies

### LoRA (Low-Rank Adaptation)
- **What**: Adds trainable low-rank matrices (A × B) to specific layers
- **Why**: Reduces trainable parameters from 8B → 8M (99% reduction)
- **Result**: Same quality, 1/10th the cost

### 4-bit Quantization
- **What**: Reduces model precision from FP16 to INT4
- **Why**: Cuts memory requirement by 75%
- **Trade-off**: Minimal accuracy loss for huge memory savings

### PEFT (Parameter Efficient Fine-Tuning)
- **What**: Framework for efficient adapters
- **Supported**: LoRA, Prefix Tuning, P-Tuning, Prompt Tuning
- **Why**: Industry standard for efficient fine-tuning

## Model Architecture

```
Input: Regulatory Q&A Data
         ↓
[Tokenization + Embedding]
         ↓
[Llama 3.1 Base Model (8B, frozen)]
         ↓
[LoRA Adapters (8M, trainable)]
         ↓
Output: Regulatory-specific responses
```

## Training Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| LoRA Rank | 8 | Rank of decomposition matrices |
| LoRA Alpha | 16 | Scaling factor (typically 2×rank) |
| Learning Rate | 2e-4 | Conservative to prevent overfitting |
| Batch Size | 4 | Limited by GPU memory |
| Gradient Accum | 4 | Effective batch size = 16 |
| Max Length | 2048 | Context window size |
| Optimizer | PagedAdamW-8bit | Memory-efficient optimizer |

## Expected Results

After training, you'll have:
1. **Specialized model** for regulatory compliance
2. **Better accuracy** on domain-specific questions
3. **Reduced hallucinations** through fine-tuning
4. **Faster inference** with smaller model memory

## Monitoring Training

### TensorBoard
```bash
# On instance
tensorboard --logdir ./llama_regulatory_model/runs

# Port forward from local
gcloud compute ssh llama-training --zone=us-central1-a \
    --ssh-flag="-L 6006:localhost:6006"

# Access http://localhost:6006
```

### GPU Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi
```

## Using Trained Model

```python
from inference_example import load_fine_tuned_model, format_prompt, generate_answer

# Load model
model, tokenizer = load_fine_tuned_model("./llama_regulatory_model")

# Ask question
question = "What are the eligibility criteria for housing finance?"
prompt = format_prompt(question)
answer = generate_answer(model, tokenizer, prompt)

print(answer)
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 2`
- Reduce max length: `--max_length 1024`

### Training Too Slow
- Increase gradient accumulation: `--gradient_accumulation_steps 8`

### Model Not Learning
- Increase epochs: `--epochs 5`
- Lower learning rate: `--learning_rate 1e-4`

## Files Structure

```
.
├── train_llama.py              # Main training script
├── inference_example.py         # Usage example
├── data_prep.py                # Data preparation
├── main.py                     # Q&A extraction
├── setup_gcp.sh                # GCP setup script
├── deploy_to_gcp.sh           # Deployment automation
├── train_data.jsonl            # Training dataset
├── val_data.jsonl              # Validation dataset
├── requirements.txt            # Dependencies
├── GCP_TRAINING_GUIDE.md       # Detailed guide
├── TRAINING_SUMMARY.md         # This file
└── llama_regulatory_model/    # Output (after training)
```

## Next Steps

1. **Train the model** on your regulatory data
2. **Evaluate** on a held-out test set
3. **Compare** with base Llama model
4. **Iterate** on hyperparameters if needed
5. **Deploy** for production use

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research
- [Hugging Face PEFT](https://huggingface.co/docs/peft) - Documentation
- [Transformers](https://huggingface.co/docs/transformers) - Model docs
- [GCP Pricing](https://cloud.google.com/compute/gpu-pricing) - Cost calculator

## Support

For questions or issues:
1. Check `GCP_TRAINING_GUIDE.md` for detailed explanations
2. Review training logs in output directory
3. Monitor GPU status with `nvidia-smi`
4. Check TensorBoard metrics for training progress
