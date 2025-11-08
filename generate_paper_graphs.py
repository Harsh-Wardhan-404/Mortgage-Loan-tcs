#!/usr/bin/env python3
"""
Generate graphs and visualizations for research paper.
Creates publication-ready figures from training data.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter
import numpy as np

# Set publication-quality style
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_trainer_state(json_path: str) -> Dict:
    """Load trainer state JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_training_metrics(trainer_state: Dict) -> Tuple[List, List, List, List, List]:
    """Extract training metrics from trainer state."""
    log_history = trainer_state.get('log_history', [])
    
    steps = []
    train_losses = []
    eval_losses = []
    learning_rates = []
    epochs = []
    
    for entry in log_history:
        if 'step' in entry:
            steps.append(entry['step'])
            epochs.append(entry.get('epoch', 0))
            
            if 'loss' in entry:
                train_losses.append(entry['loss'])
            else:
                train_losses.append(None)
            
            if 'eval_loss' in entry:
                eval_losses.append(entry['eval_loss'])
            else:
                eval_losses.append(None)
            
            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])
            else:
                learning_rates.append(None)
    
    return steps, train_losses, eval_losses, learning_rates, epochs

def plot_loss_curves(steps: List[int], train_losses: List[float], 
                     eval_losses: List[float], output_dir: str):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out None values for training loss
    train_steps = [s for s, l in zip(steps, train_losses) if l is not None]
    train_vals = [l for l in train_losses if l is not None]
    
    # Filter out None values for eval loss
    eval_steps = [s for s, l in zip(steps, eval_losses) if l is not None]
    eval_vals = [l for l in eval_losses if l is not None]
    
    ax.plot(train_steps, train_vals, label='Training Loss', linewidth=2, alpha=0.8)
    if eval_steps:
        ax.plot(eval_steps, eval_vals, label='Validation Loss', linewidth=2, 
                marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training and Validation Loss Curves', fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'loss_curves.pdf'), bbox_inches='tight')
    print(f"✓ Saved loss curves to {output_dir}/loss_curves.png/pdf")
    plt.close()

def plot_learning_rate_schedule(steps: List[int], learning_rates: List[float], 
                                output_dir: str):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out None values
    lr_steps = [s for s, lr in zip(steps, learning_rates) if lr is not None]
    lr_vals = [lr for lr in learning_rates if lr is not None]
    
    ax.plot(lr_steps, lr_vals, linewidth=2, color='#2E86AB', alpha=0.8)
    ax.set_xlabel('Training Steps', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.pdf'), bbox_inches='tight')
    print(f"✓ Saved learning rate schedule to {output_dir}/learning_rate_schedule.png/pdf")
    plt.close()

def analyze_dataset(train_file: str, val_file: str) -> Dict:
    """Analyze dataset statistics."""
    stats = {
        'train_count': 0,
        'val_count': 0,
        'train_question_lengths': [],
        'train_answer_lengths': [],
        'val_question_lengths': [],
        'val_answer_lengths': [],
    }
    
    # Analyze training data
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    stats['train_count'] += 1
                    instruction = data.get('instruction', '')
                    output = data.get('output', '')
                    stats['train_question_lengths'].append(len(instruction.split()))
                    stats['train_answer_lengths'].append(len(output.split()))
    
    # Analyze validation data
    if os.path.exists(val_file):
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    stats['val_count'] += 1
                    instruction = data.get('instruction', '')
                    output = data.get('output', '')
                    stats['val_question_lengths'].append(len(instruction.split()))
                    stats['val_answer_lengths'].append(len(output.split()))
    
    return stats

def plot_dataset_statistics(stats: Dict, output_dir: str):
    """Plot dataset statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Dataset split
    ax = axes[0, 0]
    sizes = [stats['train_count'], stats['val_count']]
    labels = ['Training', 'Validation']
    colors = ['#2E86AB', '#A23B72']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
           colors=colors, textprops={'fontweight': 'bold'})
    ax.set_title('Dataset Split', fontweight='bold', pad=15)
    
    # Question length distribution
    ax = axes[0, 1]
    ax.hist(stats['train_question_lengths'], bins=30, alpha=0.7, 
            label='Training', color='#2E86AB', edgecolor='black')
    ax.hist(stats['val_question_lengths'], bins=30, alpha=0.7, 
            label='Validation', color='#A23B72', edgecolor='black')
    ax.set_xlabel('Question Length (words)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Question Length Distribution', fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Answer length distribution
    ax = axes[1, 0]
    ax.hist(stats['train_answer_lengths'], bins=30, alpha=0.7, 
            label='Training', color='#2E86AB', edgecolor='black')
    ax.hist(stats['val_answer_lengths'], bins=30, alpha=0.7, 
            label='Validation', color='#A23B72', edgecolor='black')
    ax.set_xlabel('Answer Length (words)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Answer Length Distribution', fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    summary_data = [
        ['Metric', 'Training', 'Validation'],
        ['Total Examples', f"{stats['train_count']:,}", f"{stats['val_count']:,}"],
        ['Avg Question Length', f"{np.mean(stats['train_question_lengths']):.1f} words", 
         f"{np.mean(stats['val_question_lengths']):.1f} words"],
        ['Avg Answer Length', f"{np.mean(stats['train_answer_lengths']):.1f} words", 
         f"{np.mean(stats['val_answer_lengths']):.1f} words"],
        ['Max Question Length', f"{max(stats['train_question_lengths'])} words", 
         f"{max(stats['val_question_lengths'])} words"],
        ['Max Answer Length', f"{max(stats['train_answer_lengths'])} words", 
         f"{max(stats['val_answer_lengths'])} words"],
    ]
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax.set_title('Dataset Statistics Summary', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_statistics.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'dataset_statistics.pdf'), bbox_inches='tight')
    print(f"✓ Saved dataset statistics to {output_dir}/dataset_statistics.png/pdf")
    plt.close()

def create_training_summary(trainer_state: Dict, stats: Dict, output_dir: str):
    """Create a summary table of training parameters and results."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    best_step = trainer_state.get('best_global_step', 'N/A')
    best_metric = trainer_state.get('best_metric', 'N/A')
    total_steps = trainer_state.get('global_step', 'N/A')
    total_epochs = trainer_state.get('epoch', 'N/A')
    
    summary_data = [
        ['Training Parameter', 'Value'],
        ['Base Model', 'Llama 3.1 8B Instruct'],
        ['Fine-tuning Method', 'LoRA (Low-Rank Adaptation)'],
        ['Quantization', '4-bit (BitsAndBytes)'],
        ['Total Training Steps', f"{total_steps:,}"],
        ['Total Epochs', f"{total_epochs:.2f}"],
        ['Best Model Checkpoint', f"Step {best_step}"],
        ['Best Validation Loss', f"{best_metric:.4f}"],
        ['Training Examples', f"{stats['train_count']:,}"],
        ['Validation Examples', f"{stats['val_count']:,}"],
        ['LoRA Rank', '8'],
        ['LoRA Alpha', '16'],
        ['Learning Rate', '2e-4'],
        ['Batch Size', '1'],
        ['Gradient Accumulation Steps', '8'],
        ['Effective Batch Size', '8'],
        ['Max Sequence Length', '1024'],
        ['Optimizer', 'PagedAdamW-8bit'],
    ]
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Training Configuration and Results Summary', 
                 fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_summary.pdf'), bbox_inches='tight')
    print(f"✓ Saved training summary to {output_dir}/training_summary.png/pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate graphs for research paper')
    parser.add_argument('--trainer_state', type=str, 
                       default='llama_regulatory_model/checkpoint-717/trainer_state.json',
                       help='Path to trainer_state.json')
    parser.add_argument('--train_data', type=str, default='train_data.jsonl',
                       help='Path to training data JSONL file')
    parser.add_argument('--val_data', type=str, default='val_data.jsonl',
                       help='Path to validation data JSONL file')
    parser.add_argument('--output_dir', type=str, default='paper_graphs',
                       help='Output directory for graphs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Research Paper Graphs")
    print("=" * 60)
    
    # Load trainer state
    print("\n[1/5] Loading training metrics...")
    trainer_state = load_trainer_state(args.trainer_state)
    steps, train_losses, eval_losses, learning_rates, epochs = extract_training_metrics(trainer_state)
    print(f"   Loaded {len(steps)} training steps")
    
    # Plot loss curves
    print("\n[2/5] Generating loss curves...")
    plot_loss_curves(steps, train_losses, eval_losses, args.output_dir)
    
    # Plot learning rate schedule
    print("\n[3/5] Generating learning rate schedule...")
    plot_learning_rate_schedule(steps, learning_rates, args.output_dir)
    
    # Analyze dataset
    print("\n[4/5] Analyzing dataset statistics...")
    stats = analyze_dataset(args.train_data, args.val_data)
    plot_dataset_statistics(stats, args.output_dir)
    
    # Create training summary
    print("\n[5/5] Creating training summary...")
    create_training_summary(trainer_state, stats, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✓ All graphs generated successfully!")
    print(f"✓ Output directory: {args.output_dir}/")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - loss_curves.png/pdf")
    print("  - learning_rate_schedule.png/pdf")
    print("  - dataset_statistics.png/pdf")
    print("  - training_summary.png/pdf")

if __name__ == '__main__':
    main()

