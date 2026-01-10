"""
eval_only.py - WITH ENSEMBLE VOTING
- Uses en_question (English) only
- Outputs TSV only (no CSV needed)
- Uses ID from test dataset first column
- ENSEMBLE VOTING: Generates 3 predictions with different temperatures and votes
"""

import os
import gc
import json
import pandas as pd
import numpy as np
import re
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from ast import literal_eval
import time
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================================
# PROMPT TEMPLATES (SAME AS TRAINING - ENSURES CONSISTENCY)
# ============================================================================

PROMPT_TEMPLATES = {
    "CONTEXT_AWARE": """You are answering cultural questions about {country}.
Provide concise answers (1-3 words) based on typical cultural knowledge.

Question: {question}

Answer:""",

    "FORMAT_AWARE": """You are answering a cultural question.
Required format: {format}
Answer concisely (1-3 words).

Question: {question}

Answer:""",

    "BASIC": """You are a helpful assistant answering cultural questions.
Answer concisely (1-3 words).

Question: {question}

Answer:""",
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
# ENSEMBLE VOTING CONFIGURATION
# ============================================================================

ENSEMBLE_CONFIG = {
    "num_predictions": 3,
    "temperatures": [0.05, 0.10, 0.15],  # Conservative, Balanced, Creative
    "voting_method": "majority",  # majority, frequency-based, confidence
}


# ============================================================================
# CORRECT EVALUATION METHOD
# ============================================================================

def extract_answers_from_annotation(annotation: str) -> List[str]:
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


def clean_text(text: str) -> str:
    """Clean text for matching"""
    if not isinstance(text, str):
        return str(text).lower().strip()
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s-]', ' ', text.lower().strip())
    # Clean multiple spaces
    text = ' '.join(text.split())
    return text


def evaluate_answer_correct_method(
    predicted: str,
    acceptable_answers: List[str],
) -> Tuple[bool, str]:
    """
    CORRECT SAQ evaluation: If prediction matches ANY acceptable answer → CORRECT
    """
    
    pred_clean = clean_text(predicted)
    
    # Check against each acceptable answer
    for acceptable in acceptable_answers:
        acceptable_clean = clean_text(acceptable)
        
        # Exact match
        if pred_clean == acceptable_clean:
            return True, f"exact:{acceptable_clean}"
        
        # Contained match
        if acceptable_clean in pred_clean or pred_clean in acceptable_clean:
            return True, f"contained:{acceptable_clean}"
    
    return False, f"no_match"


# ============================================================================
# ENSEMBLE VOTING HELPER FUNCTIONS
# ============================================================================

def vote_ensemble_predictions(predictions: List[str]) -> Tuple[str, float]:
    """
    Vote on ensemble predictions and return winning answer with confidence
    
    Args:
        predictions: List of generated answers from different temperatures
    
    Returns:
        (winning_answer, confidence_score)
    """
    
    # Clean all predictions
    cleaned_preds = [pred.strip() for pred in predictions if pred]
    
    if not cleaned_preds:
        return "", 0.0
    
    if len(cleaned_preds) == 1:
        return cleaned_preds[0], 1.0
    
    # Count votes
    vote_counts = Counter(cleaned_preds)
    winning_answer, vote_count = vote_counts.most_common(1)[0]
    
    # Calculate confidence (percentage of votes received)
    confidence = vote_count / len(cleaned_preds)
    
    return winning_answer, confidence


def format_ensemble_debug(predictions: List[str], winner: str, confidence: float) -> str:
    """Format ensemble voting information for debugging"""
    debug_info = f"[Ensemble] "
    for i, pred in enumerate(predictions):
        debug_info += f"Gen{i+1}({ENSEMBLE_CONFIG['temperatures'][i]}):'{pred}' "
    debug_info += f"→ Vote:'{winner}'({confidence:.1%})"
    return debug_info


# ============================================================================
# IMPROVED EVALUATOR WITH ENSEMBLE VOTING
# ============================================================================

class ModelEvaluatorWithEnsemble:
    """Evaluate fine-tuned model using ensemble voting strategy"""
    
    def __init__(
        self,
        model_path: str = "saq/results/lora_model",
        base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token: str = None,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.hf_token = hf_token
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        
        self.load_model()
    
    def load_model(self):
        """Load fine-tuned model"""
        
        print("\n" + "="*70)
        print("LOADING FINE-TUNED MODEL")
        print("="*70)
        
        print(f"Loading tokenizer from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model: {self.base_model}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
            quantization_config=quantization_config,
            attn_implementation="eager",
        )
        
        print(f"Loading LoRA adapters from: {self.model_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_path,
            is_trainable=False,
        )
        
        self.model.eval()
        print(f"✓ Model loaded successfully")
    
    def clean_generated_answer(self, answer: str) -> str:
        """
        AGGRESSIVE cleaning to remove instruction text
        Extract ONLY the core answer
        """
        
        # Remove common prefixes first
        answer = re.sub(r'^(Answer[:]*\s*|The answer is[:]*\s*)', '', answer, flags=re.IGNORECASE)
        
        # Remove everything after parentheses (instructions in brackets)
        answer = re.split(r'[\(\[]', answer)[0].strip()
        
        # Remove everything after common instruction keywords
        instruction_keywords = [
            r'(?:Provide|provide)',
            r'(?:Give|give)',
            r'(?:Note|note)',
            r'(?:Please|please)',
            r'(?:Remember|remember)',
            r'(?:Note:|note:)',
        ]
        
        for keyword in instruction_keywords:
            match = re.search(keyword, answer, flags=re.IGNORECASE)
            if match:
                answer = answer[:match.start()].strip()
        
        # Remove trailing punctuation
        answer = answer.rstrip('.,;:!?\'" ')
        
        # If still too long (>5 words), take first 3-4 words
        words = answer.split()
        if len(words) > 4:
            answer = ' '.join(words[:3])
        
        return answer.strip()
    
    def generate_answer_single(
        self,
        question: str,
        country: str = None,
        answer_format: str = None,
        template: str = "CONTEXT_AWARE",
        temperature: float = 0.1,
        max_tokens: int = 15,
    ) -> str:
        """
        Generate SINGLE answer with given temperature
        """
        
        # Prepare prompt
        if template not in PROMPT_TEMPLATES:
            template = "CONTEXT_AWARE"
        
        prompt_template = PROMPT_TEMPLATES[template]
        country_name = COUNTRY_NAMES.get(country, "World")
        format_desc = FORMAT_NAMES.get(answer_format, "plain text")
        
        prompt = prompt_template.format(
            country=country_name,
            question=question,
            format=format_desc
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Generate with specific temperature
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=False if temperature < 0.01 else True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        
        # Clean
        answer = self.clean_generated_answer(answer)
        
        return answer
    
    def generate_answer_ensemble(
        self,
        question: str,
        country: str = None,
        answer_format: str = None,
        template: str = "CONTEXT_AWARE",
        max_tokens: int = 15,
        debug: bool = False,
    ) -> Tuple[str, float, List[str]]:
        """
        Generate ENSEMBLE of predictions using different temperatures
        Vote on best answer
        
        Returns:
            (winning_answer, confidence, all_predictions)
        """
        
        predictions = []
        temperatures = ENSEMBLE_CONFIG["temperatures"]
        
        # Generate with each temperature
        for temp in temperatures:
            pred = self.generate_answer_single(
                question=question,
                country=country,
                answer_format=answer_format,
                template=template,
                temperature=temp,
                max_tokens=max_tokens,
            )
            predictions.append(pred)
        
        # Vote on predictions
        winning_answer, confidence = vote_ensemble_predictions(predictions)
        
        if debug:
            debug_str = format_ensemble_debug(predictions, winning_answer, confidence)
            print(f"    {debug_str}")
        
        return winning_answer, confidence, predictions
    
    def evaluate_on_validation(
        self,
        val_dataset: List[Dict],
        use_ensemble: bool = True,
        debug_sample_count: int = 5,
    ) -> Dict:
        """
        Evaluate on validation set using CORRECT SAQ method
        Optionally uses ensemble voting
        """
        
        print("\n" + "="*70)
        print("VALIDATION EVALUATION (WITH ENSEMBLE VOTING)" if use_ensemble else "VALIDATION EVALUATION")
        print("="*70)
        print("\nEvaluation Logic:")
        print("  • Generate answer using ensemble voting strategy" if use_ensemble else "  • Generate answer (single)")
        print("  • Use en_question (English) only")
        print("  • If matches ANY acceptable answer → ✅ CORRECT")
        print("  • If doesn't match any → ❌ WRONG")
        
        if use_ensemble:
            print(f"\nEnsemble Config:")
            print(f"  • Num predictions: {ENSEMBLE_CONFIG['num_predictions']}")
            print(f"  • Temperatures: {ENSEMBLE_CONFIG['temperatures']}")
            print(f"  • Voting method: {ENSEMBLE_CONFIG['voting_method']}")
        
        if not val_dataset:
            print("⚠️  No validation dataset")
            return {}
        
        correct_count = 0
        start_time = time.time()
        
        print(f"\nEvaluating {len(val_dataset)} samples...")
        
        with torch.no_grad():
            for idx, sample in enumerate(val_dataset):
                try:
                    # Use en_question (English) only
                    question = sample.get('en_question', '')
                    country = sample.get('country', '')
                    answer_format = sample.get('format', '')
                    annotation = sample.get('annotations', '[]')
                    
                    if not question:
                        continue
                    
                    # Extract acceptable answers
                    acceptable_answers = extract_answers_from_annotation(annotation)
                    if not acceptable_answers:
                        acceptable_answers = [sample.get('answer', '')]
                    
                    # Generate answer (with ensemble)
                    if use_ensemble:
                        predicted, confidence, all_preds = self.generate_answer_ensemble(
                            question=question,
                            country=country,
                            answer_format=answer_format,
                            template="CONTEXT_AWARE",
                            debug=False,
                        )
                    else:
                        predicted = self.generate_answer_single(
                            question=question,
                            country=country,
                            answer_format=answer_format,
                            template="CONTEXT_AWARE",
                        )
                        confidence = 1.0
                    
                    # Evaluate
                    is_correct, match_info = evaluate_answer_correct_method(
                        predicted,
                        acceptable_answers
                    )
                    
                    if is_correct:
                        correct_count += 1
                    
                    if (idx + 1) % 20 == 0:
                        elapsed = time.time() - start_time
                        acc = correct_count / (idx + 1)
                        print(f"  {idx + 1}/{len(val_dataset)} - Accuracy: {acc:.2%} ({elapsed:.1f}s)")
                
                except Exception as e:
                    print(f"  ⚠️  Error at sample {idx}: {str(e)}")
        
        elapsed = time.time() - start_time
        accuracy = correct_count / len(val_dataset) if val_dataset else 0
        
        print(f"\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"✅ Correct: {correct_count}/{len(val_dataset)}")
        print(f"❌ Wrong: {len(val_dataset) - correct_count}/{len(val_dataset)}")
        print(f"📊 Accuracy: {accuracy:.2%}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print("="*70)
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(val_dataset),
            'eval_time': elapsed,
        }
    
    def predict_on_test(
        self,
        test_dataset: List[Dict],
        output_path: str = "saq/results/saq_prediction.tsv",
        use_ensemble: bool = True,
    ) -> pd.DataFrame:
        """
        Generate predictions for test set using ensemble voting
        Output TSV with ID from test dataset and answers
        """
        
        print("\n" + "="*70)
        print("TEST SET PREDICTION (WITH ENSEMBLE VOTING)" if use_ensemble else "TEST SET PREDICTION")
        print("="*70)
        
        if not test_dataset:
            print("❌ No test dataset")
            return None
        
        ids = []
        answers = []
        confidences = []
        
        print(f"Generating predictions for {len(test_dataset)} samples...")
        print("Using en_question (English) only")
        
        if use_ensemble:
            print(f"Using ensemble voting with {ENSEMBLE_CONFIG['num_predictions']} predictions")
        
        start_time = time.time()
        
        with torch.no_grad():
            for idx, sample in enumerate(test_dataset):
                try:
                    # Use en_question (English) only
                    question = sample.get('en_question', '')
                    country = sample.get('country', '')
                    answer_format = sample.get('format', '')
                    # Get ID from test dataset first column
                    sample_id = sample.get('id', f'test_{idx}')
                    
                    if not question:
                        ids.append(sample_id)
                        answers.append("ERROR_NO_QUESTION")
                        confidences.append(0.0)
                        continue
                    
                    # Generate answer with ensemble
                    if use_ensemble:
                        answer, confidence, all_preds = self.generate_answer_ensemble(
                            question=question,
                            country=country,
                            answer_format=answer_format,
                            template="CONTEXT_AWARE",
                            debug=False,
                        )
                    else:
                        answer = self.generate_answer_single(
                            question=question,
                            country=country,
                            answer_format=answer_format,
                            template="CONTEXT_AWARE",
                        )
                        confidence = 1.0
                    
                    ids.append(sample_id)
                    answers.append(answer)
                    confidences.append(confidence)
                    
                    if (idx + 1) % 30 == 0:
                        elapsed = time.time() - start_time
                        print(f"  {idx + 1}/{len(test_dataset)} ({elapsed:.1f}s)")
                
                except Exception as e:
                    print(f"  ⚠️  Error at sample {idx}: {str(e)}")
                    ids.append(sample.get('id', f'test_{idx}'))
                    answers.append("ERROR")
                    confidences.append(0.0)
        
        elapsed = time.time() - start_time
        print(f"✓ Predictions completed in {elapsed:.1f}s")
        
        # Save TSV
        tsv_df = pd.DataFrame({
            'ID': ids,
            'answer': answers,
        })
        
        # Optional: Save detailed results with confidence
        detailed_df = pd.DataFrame({
            'ID': ids,
            'answer': answers,
            'confidence': confidences,
        })
        
        Path("results").mkdir(exist_ok=True)
        tsv_df.to_csv(output_path, sep='\t', index=False, header=True)
        detailed_path = output_path.replace('.tsv', '_detailed.tsv')
        detailed_df.to_csv(detailed_path, sep='\t', index=False, header=True)
        
        print(f"\n✓ TSV saved: {output_path}")
        print(f"✓ Detailed results saved: {detailed_path}")
        print(f"  Total predictions: {len(tsv_df)}")
        print(f"  Avg confidence: {np.mean(confidences):.2%}")
        
        # Show sample
        print(f"\nSample predictions (first 5):")
        print("ID\tanswer\tconfidence")
        for i in range(min(5, len(tsv_df))):
            ans = tsv_df.iloc[i]['answer'][:50]
            conf = confidences[i]
            print(f"{tsv_df.iloc[i]['ID']}\t{ans}\t{conf:.1%}")
        
        return tsv_df


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """Main evaluation with ensemble voting"""
    
    print("\n" + "="*70)
    print("🚀 EVALUATION WITH ENSEMBLE VOTING")
    print("="*70)
    print("\nStrategy:")
    print("  ✅ Uses en_question (English) only")
    print("  ✅ Uses ensemble voting with 3 different temperatures")
    print("  ✅ Generates SHORT, CRISP answers (1-3 words)")
    print("  ✅ Uses CORRECT SAQ evaluation method")
    print("  ✅ Outputs TSV + detailed results with confidence")
    print("  ✅ AGGRESSIVE cleaning of instruction text")
    
    # Check model
    if not Path("saq/results/lora_model").exists():
        print("\n❌ Trained model not found at saq/results/lora_model")
        exit(1)
    
    print("\n✓ Found trained model")
    
    # Load data
    print("\nLoading data...")
    
    train_df = pd.read_csv("saq/data/train_dataset_saq.csv")
    test_df = pd.read_csv("saq/data/test_dataset_saq.csv")
    
    # Validation split
    val_split = 0.2
    split_idx = int(len(train_df) * (1 - val_split))
    val_data = train_df.iloc[split_idx:].to_dict('records')
    
    # Test data with proper ID
    test_data = test_df.to_dict('records')
    for i, record in enumerate(test_data):
        if 'ID' in record:
            record['id'] = record['ID']
    
    print(f"✓ Loaded {len(val_data)} validation samples")
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Initialize evaluator
    evaluator = ModelEvaluatorWithEnsemble(
        model_path="saq/results/lora_model",
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token=HF_TOKEN,
    )
    
    # Evaluate validation WITH ENSEMBLE
    eval_results = evaluator.evaluate_on_validation(val_data, use_ensemble=True)
    
    # Predict test WITH ENSEMBLE
    test_predictions = evaluator.predict_on_test(
        test_data,
        output_path="saqresults/saq_prediction.tsv",
        use_ensemble=True
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ COMPLETED")
    print("="*70)
    print(f"\nValidation Accuracy: {eval_results.get('accuracy', 0):.2%}")
    print(f"Output files:")
    print(f"  • saq/results/saq_prediction.tsv (main TSV)")
    print(f"  • saq/results/saq_prediction_detailed.tsv (with confidence scores)")
    print(f"\nEnsemble settings:")
    print(f"  • Temperatures: {ENSEMBLE_CONFIG['temperatures']}")
    print(f"  • Voting method: {ENSEMBLE_CONFIG['voting_method']}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()