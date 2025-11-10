"""
Example: Using your fine-tuned Llama model for regulatory Q&A

This script demonstrates how to load and use your fine-tuned model
for generating regulatory compliance answers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def load_fine_tuned_model(model_path: str, base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Load the fine-tuned model with LoRA adapters.
    
    Args:
        model_path: Path to fine-tuned model directory
        base_model_name: Name of the base model
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        # Use 4-bit quantization on GPU
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded with 4-bit quantization on GPU")
    else:
        # CPU mode: Load in FP16 or FP32 (no quantization)
        print("CUDA not available. Loading model in FP16 for CPU inference...")
        print("Note: This will use significant RAM (~16GB). Consider using a smaller model or running on GPU.")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("Model loaded in FP16 on CPU")
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer


def format_prompt(question: str, context: str = None) -> str:
    """
    Format a regulatory question for the model.
    
    Args:
        question: The question to ask
        context: Optional context or source information
    
    Returns:
        Formatted prompt string
    """
    if context:
        prompt = f"""### Instruction:
You are a compliance analyst. Answer regulatory questions based on the provided context.

### Input:
Context: {context}

Question: {question}

### Response:
"""
    else:
        prompt = f"""### Instruction:
You are a compliance analyst. Answer regulatory questions based on your training.

### Input:
Question: {question}

### Response:
"""
    return prompt


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """
    Generate an answer using the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: The formatted prompt
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated answer string
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to device (CPU or CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    response_start = full_response.find("### Response:\n")
    if response_start != -1:
        answer = full_response[response_start + len("### Response:\n"):].strip()
    else:
        answer = full_response[len(prompt):].strip()
    
    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Use fine-tuned Llama for regulatory Q&A"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./llama_regulatory_model",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name (default: Llama 3.1 8B Instruct)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (or will be prompted)"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context for the question"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_fine_tuned_model(args.model_path, args.base_model)
    print("Model loaded successfully!\n")
    
    # Get question
    if args.question:
        question = args.question
    else:
        print("Enter your regulatory compliance question (or 'quit' to exit):")
        question = input("> ")
        if question.lower() == 'quit':
            return
    
    # Generate answer
    prompt = format_prompt(question, args.context)
    answer = generate_answer(model, tokenizer, prompt, args.max_tokens)
    
    print("\n" + "="*80)
    print("Question:")
    print(question)
    print("\nAnswer:")
    print(answer)
    print("="*80)
    
    # Interactive mode
    while True:
        print("\nEnter another question (or 'quit' to exit):")
        question = input("> ")
        if question.lower() == 'quit':
            break
        
        prompt = format_prompt(question, args.context)
        answer = generate_answer(model, tokenizer, prompt, args.max_tokens)
        
        print("\n" + "="*80)
        print("Question:")
        print(question)
        print("\nAnswer:")
        print(answer)
        print("="*80)


if __name__ == "__main__":
    main()
