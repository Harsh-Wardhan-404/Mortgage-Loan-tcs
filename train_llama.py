"""
Fine-tune Llama model on regulatory compliance data using LoRA for efficient training.

This script:
1. Loads Llama model with 8-bit quantization for memory efficiency
2. Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
3. Trains on your regulatory QnA dataset
4. Saves checkpoints during training
5. Supports GCP environment with GPU resources
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from accelerate import Accelerator


class RegulatoryDataset(Dataset):
    """Custom dataset for regulatory compliance QnA data."""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load JSONL file
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Construct the training prompt
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # Format as instruction-following text
        full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}</s>"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def setup_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3-8B-Instruct",
    use_4bit: bool = True,
    use_nested_quant: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    load_in_4bit: bool = True,
    device_map: str = "auto"
) -> tuple:
    """
    Setup model with 4-bit/8-bit quantization and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        use_4bit: Enable 4-bit quantization
        use_nested_quant: Enable nested quantization
        bnb_4bit_compute_dtype: Compute dtype (float16 or bfloat16)
        bnb_4bit_quant_type: Quantization type (fp4 or nf4)
        load_in_4bit: Load model in 4-bit
        device_map: Device mapping strategy
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Setup quantization config
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if load_in_4bit else None,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def setup_lora(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    target_modules: list = None
) -> None:
    """
    Configure LoRA adapters for efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) enables training large models with minimal parameters.
    It adds trainable rank-decomposition matrices while keeping the base model frozen.
    
    Args:
        model: The model to configure
        r: LoRA rank (lower = fewer parameters)
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: LoRA dropout rate
        bias: Bias training type
        task_type: Task type for PEFT
        target_modules: Modules to apply LoRA to (auto-detect if None)
    """
    if target_modules is None:
        # Default target modules for Llama
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama model on regulatory compliance data"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="train_data.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="val_data.jsonl",
        help="Path to validation JSONL file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Hugging Face model name (default: Llama 3 8B Instruct)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./llama_regulatory_model",
        help="Output directory for model checkpoints"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision training"
    )
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not os.path.exists(args.train_data):
        print(f"Error: Training data file not found: {args.train_data}")
        sys.exit(1)
    
    if not os.path.exists(args.val_data):
        print(f"Error: Validation data file not found: {args.val_data}")
        sys.exit(1)
    
    print("=" * 80)
    print("Llama Regulatory Compliance Fine-tuning")
    print("=" * 80)
    
    # Setup device and model
    print("\n[1/5] Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        model_name=args.model_name,
        bnb_4bit_compute_dtype="float16" if args.fp16 else "bfloat16",
    )
    
    # Setup LoRA
    print("\n[2/5] Configuring LoRA adapters...")
    model = setup_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=None  # Auto-detect
    )
    
    # Load datasets
    print("\n[3/5] Loading datasets...")
    train_dataset = RegulatoryDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = RegulatoryDataset(args.val_data, tokenizer, args.max_length)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    print("\n[4/5] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # Use 8-bit optimizer
    )
    
    # Initialize trainer
    print("\n[5/5] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 80)
    print(f"Training complete! Model saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
