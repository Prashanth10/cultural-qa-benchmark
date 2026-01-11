"""
train_lora_with_prompting_v2_CORRECTED.py

CORRECTED VERSION - Uses en_question only with proper prompting strategy
This file contains the COMPLETE, WORKING training code
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from ast import literal_eval
import logging

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT TEMPLATES (FROM BASELINE - PROVEN STRATEGY)
# ============================================================================

PROMPT_TEMPLATES = {
    "CONTEXT_AWARE": """You are answering cultural questions about {country}.
Provide concise answers (1-3 words) based on typical cultural knowledge.

Question: {question}

Answer: {answer}""",

    "FORMAT_AWARE": """You are answering a cultural question.
Required format: {format}
Answer concisely (1-3 words).

Question: {question}

Answer: {answer}""",

    "BASIC": """You are a helpful assistant answering cultural questions.
Answer concisely (1-3 words).

Question: {question}

Answer: {answer}""",
}

COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "CN": "China",
    "IR": "Iran"
}

FORMAT_NAMES = {
    "TEXT": "plain text (1-3 words)",
    "HHMM": "time format (HH:MM)",
    "NUMERIC": "numeric",
}


# ============================================================================
# DATA PREPARATION WITH PROMPTING STRATEGY
# ============================================================================

def extract_acceptable_answers(annotation: str) -> List[str]:
    """Extract all acceptable answers from annotation JSON"""
    try:
        if isinstance(annotation, str):
            annotations = literal_eval(annotation)
        else:
            annotations = annotation
        
        acceptable_answers = []
        for item in annotations:
            if 'en_answers' in item and item['en_answers']:
                acceptable_answers.extend(item['en_answers'])
            else:
                acceptable_answers.extend(item.get('answers', []))
        
        return acceptable_answers
    except:
        return []


def select_best_answer(acceptable_answers: List[str]) -> str:
    """
    Select the best answer from acceptable answers.
    Prefer shorter answers for conciseness.
    """
    if not acceptable_answers:
        return ""
    
    # Filter to get shortest answers (1-3 words)
    short_answers = []
    for ans in acceptable_answers:
        word_count = len(ans.split())
        if word_count <= 3:
            short_answers.append((ans, word_count))
    
    if short_answers:
        # Sort by word count, then by frequency
        short_answers.sort(key=lambda x: (x[1], acceptable_answers.index(x[0])))
        return short_answers[0][0]
    
    # Fallback to shortest answer overall
    return min(acceptable_answers, key=lambda x: len(x.split()))


def create_training_prompt(
    row: Dict,
    template: str = "CONTEXT_AWARE"
) -> str:
    """
    Create training prompt with answer using baseline strategy
    Uses en_question (English only) - NO translation columns
    """
    # Use en_question (English version) only
    question = row.get('en_question', '')
    country = row.get('country', '')
    answer_format = row.get('format', '')
    annotation = row.get('annotations', '')
    
    if not question:
        return None
    
    # Get best answer (short and crisp)
    acceptable_answers = extract_acceptable_answers(annotation)
    best_answer = select_best_answer(acceptable_answers)
    
    if not best_answer:
        return None
    
    # Get country and format names
    country_name = COUNTRY_NAMES.get(country, "World")
    format_desc = FORMAT_NAMES.get(answer_format, "plain text")
    
    # Select template
    if template not in PROMPT_TEMPLATES:
        template = "CONTEXT_AWARE"
    
    prompt_template = PROMPT_TEMPLATES[template]
    
    try:
        prompt = prompt_template.format(
            country=country_name,
            question=question,
            answer=best_answer,
            format=format_desc
        )
        return prompt
    except Exception as e:
        logger.warning(f"Error formatting prompt: {e}")
        return None


def prepare_dataset(
    csv_path: str,
    max_samples: Optional[int] = None,
    val_split: float = 0.2,
) -> tuple:
    """
    Prepare dataset with prompting strategy
    Uses en_question only
    """
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    logger.info(f"Total samples: {len(df)}")
    
    # Create training examples with prompts
    logger.info("Creating training prompts (using en_question only)...")
    
    training_examples = []
    skipped = 0
    
    for idx, row in df.iterrows():
        # Try all templates to find good ones
        for template in ["CONTEXT_AWARE", "FORMAT_AWARE", "BASIC"]:
            prompt = create_training_prompt(row, template=template)
            
            if prompt and len(prompt) > 20:
                training_examples.append({
                    'text': prompt,
                    'en_question': row.get('en_question', ''),
                    'country': row.get('country', ''),
                    'template': template,
                })
            else:
                skipped += 1
    
    logger.info(f"✓ Created {len(training_examples)} training examples (skipped {skipped})")
    
    # Split into train/val
    split_idx = int(len(training_examples) * (1 - val_split))
    
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        'text': [ex['text'] for ex in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        'text': [ex['text'] for ex in val_data]
    })
    
    return train_dataset, val_dataset


# ============================================================================
# LORA FINE-TUNING
# ============================================================================

class LoRATrainer:
    """Fine-tune Llama with LoRA + Prompting Strategy"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        output_dir: str = "saq/results/lora_model",
        hf_token: str = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.hf_token = hf_token
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self):
        """Load base model and tokenizer"""
        
        logger.info("=" * 70)
        logger.info("PHASE 1: LOADING MODEL AND TOKENIZER")
        logger.info("=" * 70)
        
        logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.hf_token,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model {self.model_name}")
        
        # 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
            quantization_config=quantization_config,
            attn_implementation="eager",
        )
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
    
    def setup_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        """Setup LoRA configuration"""
        
        logger.info("=" * 70)
        logger.info("PHASE 2: SETTING UP LORA")
        logger.info("=" * 70)
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj"],
        )
        
        logger.info(f"LoRA Configuration:")
        logger.info(f"  r: {r}")
        logger.info(f"  alpha: {lora_alpha}")
        logger.info(f"  dropout: {lora_dropout}")
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        self.model.print_trainable_parameters()
    
    def tokenize_function(self, examples: Dict):
        """Tokenize examples"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,              
        gradient_accumulation_steps=2,   
        weight_decay=0.01, 
    ):
        """Train with LoRA"""
        
        logger.info("=" * 70)
        logger.info("PHASE 3: TOKENIZING DATA")
        logger.info("=" * 70)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text'],
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text'],
        )
        
        logger.info(f"✓ Data tokenized")
        logger.info(f"  Train: {len(train_dataset)}")
        logger.info(f"  Val: {len(val_dataset)}")
        
        logger.info("=" * 70)
        logger.info("PHASE 4: TRAINING")
        logger.info("=" * 70)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            push_to_hub=False,
            seed=42,
            fp16=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        logger.info(f"Training Configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        logger.info(f"✓ Training completed")
        
        # Save model
        logger.info("=" * 70)
        logger.info("PHASE 5: SAVING MODEL")
        logger.info("=" * 70)
        
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"✓ Model saved to {self.output_dir}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Full training pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 LORA FINE-TUNING WITH PROMPTING STRATEGY (CORRECTED v2)")
    print("=" * 70)
    print("\nStrategy:")
    print("  ✅ Uses en_question (English) only")
    print("  ✅ Uses proven prompt templates from baseline")
    print("  ✅ Trains to generate SHORT answers (1-3 words)")
    print("  ✅ Context-aware prompts with country information")
    print("  ✅ Saves trained adapters for inference")
    
    # Prepare data
    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)
    
    train_dataset, val_dataset = prepare_dataset(
        csv_path="saq/data/train_dataset_saq.csv",
        max_samples=None,  # Use all data
        val_split=0.2,
    )
    
    # Initialize trainer
    trainer = LoRATrainer(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        output_dir="saq/results/lora_model",
        hf_token=HF_TOKEN,
    )
    
    # Load model
    trainer.load_model()
    
    # Setup LoRA
    trainer.setup_lora(r=16, lora_alpha=32, lora_dropout=0.05)
    
    # Train
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        warmup_steps=100,                
        gradient_accumulation_steps=2,   
        weight_decay=0.01, 
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run evaluation: python eval_only.py")
    print("  2. Submit: saq/results/saq_prediction.tsv")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()