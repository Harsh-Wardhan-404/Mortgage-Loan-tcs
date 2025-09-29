# data_preparation.py
import json
import random
from typing import List, Dict, Any

def convert_to_instruction_format(json_file: str, output_file: str):
    """Convert QnA JSON to instruction-following format for fine-tuning."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single object and list of objects
    if isinstance(data, dict):
        data = [data]
    
    training_examples = []
    
    for doc in data:
        source_path = doc.get('metadata', {}).get('source_path', '')
        qna_list = doc.get('qna', [])
        
        for qna in qna_list:
            question = qna.get('question', '').strip()
            answer = qna.get('answer', '').strip()
            citations = qna.get('citations', [])
            
            if not question or not answer:
                continue
            
            # Create instruction prompt
            instruction = f"""You are a compliance analyst. Extract structured QnA JSON from regulatory documents.

Focus on:
- Enforceable obligations, eligibility, disclosures, KYC/AML, risk, timelines, penalties, exceptions
- Mortgage-related clauses and regulatory requirements
- Cite sections/pages verbatim where possible

Return ONLY valid JSON with this exact structure:
{{
  "metadata": {{
    "source_path": "string",
    "doc_type": "regulatory", 
    "language": "string",
    "notes": "string"
  }},
  "qna": [
    {{
      "question": "string",
      "answer": "string", 
      "citations": ["string"]
    }}
  ]
}}

Document source: {source_path}"""
            
            # Create the target JSON
            target_json = {
                "metadata": {
                    "source_path": source_path,
                    "doc_type": "regulatory",
                    "language": "en",
                    "notes": ""
                },
                "qna": [{
                    "question": question,
                    "answer": answer,
                    "citations": citations
                }]
            }
            
            training_examples.append({
                "instruction": instruction,
                "input": f"Extract QnA from: {source_path}",
                "output": json.dumps(target_json, ensure_ascii=False)
            })
    
    # Shuffle and split
    random.shuffle(training_examples)
    split_idx = int(0.9 * len(training_examples))
    
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    # Save training data
    with open('train_data.jsonl', 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Save validation data
    with open('val_data.jsonl', 'w', encoding='utf-8') as f:
        for example in val_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Created {len(train_data)} training examples and {len(val_data)} validation examples")
    return len(train_data), len(val_data)

# Run the conversion
convert_to_instruction_format('rbi_housing_finance_qna.json', 'training_data.jsonl')
