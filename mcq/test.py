import csv
import json
import re
import sys
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
print("HF_TOKEN loaded:", os.getenv("HF_TOKEN")[:8])
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_csv(path):
    """Reads CSV and returns a list of dictionaries."""
    data = []
    with open(path, mode='r', encoding='utf-8') as f:
        # DictReader uses the first row of your CSV as keys for the dictionary
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def save_json(obj, path):
    """Saves the final predictions to a JSON file."""
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def build_prompt(q):
    """
    Constructs the prompt. 
    Assumes CSV headers: 'question', 'A', 'B', 'C', 'D'
    """
    return (
        f"Question: {q['question']}\n"
        f"A. {q['A']}\n"
        f"B. {q['B']}\n"
        f"C. {q['C']}\n"
        f"D. {q['D']}\n\n"
        "Answer with only one letter: A, B, C, or D.\nAnswer:"
    )

def extract_choice(text):
    """Picks the first A, B, C, or D found in the model output."""
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else "A"  # Fallback to A if no choice found

def main():
    # --- Configuration ---
    in_path = "cultural-qa-benchmark/mcq/data/test_dataset_mcq.csv"  # Ensure this file exists
    out_path = "pred_mcq.json"

    if not Path(in_path).exists():
        print(f"Error: Could not find input file at {in_path}")
        return

    print(f"Loading data from {in_path}...")
    data = load_csv(in_path)

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # Using device_map="auto" to automatically handle GPU placement on HPC
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    preds = []
    print("Starting inference...")
    
    for i, item in enumerate(data):
        try:
            prompt = build_prompt(item)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract only the newly generated text (the answer)
            tail = decoded[len(prompt):]
            choice = extract_choice(tail)

            # Use 'id' from CSV if it exists, otherwise use the index
            record_id = item.get("id", i)
            preds.append({"id": record_id, "prediction": choice})
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)} items...")
        
        except Exception as e:
            print(f"Error processing item {i}: {e}")

    save_json(preds, out_path)
    print(f"Successfully saved {len(preds)} predictions to {out_path}")

if __name__ == "__main__":
    main()