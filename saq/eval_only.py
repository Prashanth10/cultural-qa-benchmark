"""
eval_only.py WITH ANALYSIS - KEEPS YOUR CODE + ADDS analysis_predictions.csv GENERATION
- Your existing ensemble voting code (unchanged)
- NEW: Analysis CSV showing predictions vs correct answers
- Useful for debugging what model gets wrong before Phase 2
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
from datetime import datetime
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
# ENSEMBLE VOTING CONFIGURATION (WITH FIXES)
# ============================================================================

ENSEMBLE_CONFIG = {
    "num_predictions": 3,
    "temperatures": [0.01, 0.10, 0.30],
    "voting_method": "majority",
}

print("""
╔════════════════════════════════════════════════════════════╗
║ ENSEMBLE VOTING WITH ANALYSIS GENERATION                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║ FIX 1: Sampling enabled (do_sample=True always)           ║
║ FIX 2: Temperature spread [0.01, 0.10, 0.30]              ║
║ FIX 3: Answer length limit 6 words (was 4)                ║
║                                                            ║
║ NEW: Creates analysis_predictions.csv with:               ║
║      - All 3 ensemble predictions                         ║
║      - Final prediction + confidence                      ║
║      - Correct answers from dataset                       ║
║      - Whether prediction was correct/wrong               ║
║                                                            ║
║ Expected improvement: +5-10% over baseline                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")

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
    text = re.sub(r'[^\w\s-]', ' ', text.lower().strip())
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
    for acceptable in acceptable_answers:
        acceptable_clean = clean_text(acceptable)
        if pred_clean == acceptable_clean:
            return True, f"exact:{acceptable_clean}"
        if acceptable_clean in pred_clean or pred_clean in acceptable_clean:
            return True, f"contained:{acceptable_clean}"
    return False, f"no_match"

# ============================================================================
# ENSEMBLE VOTING HELPER FUNCTIONS
# ============================================================================

def vote_ensemble_predictions(predictions: List[str]) -> Tuple[str, float]:
    """
    Vote on ensemble predictions and return winning answer with confidence
    """
    cleaned_preds = [pred.strip() for pred in predictions if pred]
    if not cleaned_preds:
        return "", 0.0
    if len(cleaned_preds) == 1:
        return cleaned_preds[0], 1.0
    
    vote_counts = Counter(cleaned_preds)
    winning_answer, vote_count = vote_counts.most_common(1)[0]
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
# IMPROVED EVALUATOR WITH ENSEMBLE VOTING (WITH FIXES)
# ============================================================================

class ModelEvaluatorWithEnsemble:
    """Evaluate fine-tuned model using ensemble voting strategy with fixes"""
    
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
        """AGGRESSIVE cleaning to remove instruction text"""
        answer = re.sub(r'^(Answer[:]*\s*|The answer is[:]*\s*)', '', answer, flags=re.IGNORECASE)
        answer = re.split(r'[\(\[]', answer)[0].strip()
        
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
        
        answer = answer.rstrip('.,;:!?\'" ')
        words = answer.split()
        if len(words) > 6:
            answer = ' '.join(words[:6])
        
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
        """Generate SINGLE answer with given temperature"""
        
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
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        
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
        """Generate ENSEMBLE of predictions using different temperatures"""
        
        predictions = []
        temperatures = ENSEMBLE_CONFIG["temperatures"]
        
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
        
        winning_answer, confidence = vote_ensemble_predictions(predictions)
        
        if debug:
            debug_str = format_ensemble_debug(predictions, winning_answer, confidence)
            print(f" {debug_str}")
        
        return winning_answer, confidence, predictions
    
    def evaluate_on_validation(
        self,
        val_dataset: List[Dict],
        use_ensemble: bool = True,
    ) -> Dict:
        """Evaluate on validation set using CORRECT SAQ method"""
        
        print("\n" + "="*70)
        print("VALIDATION EVALUATION (WITH ENSEMBLE VOTING - FIXES APPLIED)")
        print("="*70)
        print("\nEvaluation Logic:")
        print(" • Generate answer using ensemble voting strategy")
        print(" • Use en_question (English) only")
        print(" • If matches ANY acceptable answer → ✅ CORRECT")
        print(" • If doesn't match any → ❌ WRONG")
        
        if use_ensemble:
            print(f"\nEnsemble Config (WITH FIXES):")
            print(f" • Num predictions: {ENSEMBLE_CONFIG['num_predictions']}")
            print(f" • Temperatures: {ENSEMBLE_CONFIG['temperatures']} (FIX: More spread)")
            print(f" • Sampling: Enabled (FIX: Always True)")
            print(f" • Max answer words: 6 (FIX: Was 4)")
            print(f" • Voting method: {ENSEMBLE_CONFIG['voting_method']}")
        
        if not val_dataset:
            print("⚠️ No validation dataset")
            return {}
        
        correct_count = 0
        start_time = time.time()
        
        print(f"\nEvaluating {len(val_dataset)} samples...")
        
        with torch.no_grad():
            for idx, sample in enumerate(val_dataset):
                try:
                    question = sample.get('en_question', '')
                    country = sample.get('country', '')
                    answer_format = sample.get('format', '')
                    annotation = sample.get('annotations', '[]')
                    
                    if not question:
                        continue
                    
                    acceptable_answers = extract_answers_from_annotation(annotation)
                    if not acceptable_answers:
                        acceptable_answers = [sample.get('answer', '')]
                    
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
                    
                    is_correct, match_info = evaluate_answer_correct_method(
                        predicted,
                        acceptable_answers
                    )
                    
                    if is_correct:
                        correct_count += 1
                    
                    if (idx + 1) % 20 == 0:
                        elapsed = time.time() - start_time
                        acc = correct_count / (idx + 1)
                        print(f" {idx + 1}/{len(val_dataset)} - Accuracy: {acc:.2%} ({elapsed:.1f}s)")
                
                except Exception as e:
                    print(f" ⚠️ Error at sample {idx}: {str(e)}")
        
        elapsed = time.time() - start_time
        accuracy = correct_count / len(val_dataset) if val_dataset else 0
        
        print(f"\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"✅ Correct: {correct_count}/{len(val_dataset)}")
        print(f"❌ Wrong: {len(val_dataset) - correct_count}/{len(val_dataset)}")
        print(f"📊 Accuracy: {accuracy:.2%}")
        print(f"⏱️ Time: {elapsed:.1f}s")
        print("="*70)
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(val_dataset),
            'eval_time': elapsed,
        }
    
    def create_analysis_predictions(
        self,
        val_dataset: List[Dict],
        output_path: str = "saq/results/analysis_predictions.csv",
    ) -> pd.DataFrame:
        """
        Create detailed analysis predictions CSV showing:
        - All 3 ensemble predictions (temp 0.01, 0.10, 0.30)
        - Final voted prediction
        - Confidence score
        - Correct answers from dataset
        - Whether prediction was correct or wrong
        """
        
        print("\n" + "="*70)
        print("CREATING DETAILED ANALYSIS CSV")
        print("="*70)
        print(f"Analyzing {len(val_dataset)} validation samples...")
        print("This creates: saq/results/analysis_predictions.csv\n")
        
        results = []
        correct_count = 0
        
        with torch.no_grad():
            for idx, sample in enumerate(val_dataset):
                try:
                    question = sample.get('en_question', '')
                    country = sample.get('country', '')
                    answer_format = sample.get('format', '')
                    annotation = sample.get('annotations', '[]')
                    
                    if not question:
                        continue
                    
                    acceptable = extract_answers_from_annotation(annotation)
                    if not acceptable:
                        acceptable = [sample.get('answer', '')]
                    
                    predicted, confidence, all_preds = self.generate_answer_ensemble(
                        question=question,
                        country=country,
                        answer_format=answer_format,
                        template="CONTEXT_AWARE",
                        debug=False,
                    )
                    
                    is_correct, match_info = evaluate_answer_correct_method(predicted, acceptable)
                    if is_correct:
                        correct_count += 1
                    
                    result = {
                        'index': idx,
                        'question': question,
                        'country': country,
                        'prediction_1': all_preds[0],
                        'prediction_2': all_preds[1],
                        'prediction_3': all_preds[2],
                        'final_prediction': predicted,
                        'confidence': f"{confidence:.1%}",
                        'acceptable_answers': '|'.join(acceptable),
                        'is_correct': 'YES' if is_correct else 'NO',
                        'correct_answer': acceptable[0] if acceptable else 'N/A',
                    }
                    results.append(result)
                    
                    if (idx + 1) % 50 == 0:
                        acc = correct_count / (idx + 1)
                        print(f"  {idx + 1}/{len(val_dataset)} - Current Accuracy: {acc:.2%}")
                
                except Exception as e:
                    print(f"  ⚠️  Error at sample {idx}: {str(e)}")
        
        df_results = pd.DataFrame(results)
        Path("test").mkdir(exist_ok=True)
        df_results.to_csv(output_path, index=False)
        
        overall_acc = correct_count / len(results) if results else 0
        
        print(f"\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total samples: {len(results)}")
        print(f"Correct: {correct_count}/{len(results)}")
        print(f"Accuracy: {overall_acc:.2%}")
        print(f"Incorrect: {len(results) - correct_count}/{len(results)}")
        
        if 'country' in df_results.columns:
            print("\nAccuracy by Country:")
            for country in sorted(df_results['country'].unique()):
                country_data = df_results[df_results['country'] == country]
                acc = (country_data['is_correct'] == 'YES').sum() / len(country_data)
                print(f"  {country}: {acc:.2%} ({(country_data['is_correct'] == 'YES').sum()}/{len(country_data)})")
        
        print("\nConfidence Analysis:")
        df_results['conf_float'] = df_results['confidence'].str.rstrip('%').astype(float) / 100
        
        correct_conf = df_results[df_results['is_correct'] == 'YES']['conf_float'].mean()
        incorrect_conf = df_results[df_results['is_correct'] == 'NO']['conf_float'].mean()
        
        print(f"  Avg confidence when correct: {correct_conf:.1%}")
        print(f"  Avg confidence when incorrect: {incorrect_conf:.1%}")
        print(f"  Difference: {correct_conf - incorrect_conf:.1%}")
        
        print("\n" + "="*70)
        print("FIRST 10 INCORRECT PREDICTIONS (FOR ANALYSIS)")
        print("="*70 + "\n")
        
        incorrect = df_results[df_results['is_correct'] == 'NO'].head(10)
        for row_idx, row in incorrect.iterrows():
            print(f"Q{row['index']+1}: {row['question'][:60]}")
            print(f"  Country: {row['country']}")
            print(f"  Predictions: {row['prediction_1']} | {row['prediction_2']} | {row['prediction_3']}")
            print(f"  Final: '{row['final_prediction']}' (Confidence: {row['confidence']})")
            print(f"  Correct: {row['correct_answer']}")
            print()
        
        print("="*70)
        print(f"✓ Saved: {output_path}")
        print("✓ Open in Excel/Sheets to analyze patterns")
        print("="*70 + "\n")
        
        return df_results
    
    def predict_on_test(
        self,
        test_dataset: List[Dict],
        output_path: str = "saq/results/saq_prediction.tsv",
        use_ensemble: bool = True,
    ) -> pd.DataFrame:
        """Generate predictions for test set using ensemble voting"""
        
        print("\n" + "="*70)
        print("TEST SET PREDICTION (WITH ENSEMBLE VOTING - FIXES APPLIED)")
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
            print(f"Temperatures: {ENSEMBLE_CONFIG['temperatures']}")
        
        start_time = time.time()
        
        with torch.no_grad():
            for idx, sample in enumerate(test_dataset):
                try:
                    question = sample.get('en_question', '')
                    country = sample.get('country', '')
                    answer_format = sample.get('format', '')
                    sample_id = sample.get('id', f'test_{idx}')
                    
                    if not question:
                        ids.append(sample_id)
                        answers.append("ERROR_NO_QUESTION")
                        confidences.append(0.0)
                        continue
                    
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
                        print(f" {idx + 1}/{len(test_dataset)} ({elapsed:.1f}s)")
                
                except Exception as e:
                    print(f" ⚠️ Error at sample {idx}: {str(e)}")
                    ids.append(sample.get('id', f'test_{idx}'))
                    answers.append("ERROR")
                    confidences.append(0.0)
        
        elapsed = time.time() - start_time
        print(f"✓ Predictions completed in {elapsed:.1f}s")
        
        tsv_df = pd.DataFrame({
            'ID': ids,
            'answer': answers,
        })
        
        detailed_df = pd.DataFrame({
            'ID': ids,
            'answer': answers,
            'confidence': confidences,
        })
        
        Path("saq/results").mkdir(exist_ok=True)
        tsv_df.to_csv(output_path, sep='\t', index=False, header=True)
        detailed_path = output_path.replace('.tsv', '_detailed.tsv')
        detailed_df.to_csv(detailed_path, sep='\t', index=False, header=True)
        
        print(f"\n✓ TSV saved: {output_path}")
        print(f"✓ Detailed results saved: {detailed_path}")
        print(f" Total predictions: {len(tsv_df)}")
        print(f" Avg confidence: {np.mean(confidences):.2%}")
        
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
    """Main evaluation with ensemble voting (WITH FIXES)"""
    
    print("\n" + "="*70)
    print("🚀 EVALUATION WITH ENSEMBLE VOTING + ANALYSIS GENERATION")
    print("="*70)
    print("\nStrategy:")
    print(" ✅ Uses en_question (English) only")
    print(" ✅ Uses ensemble voting with 3 different temperatures")
    print(" ✅ FIX 1: Sampling enabled (do_sample=True always)")
    print(" ✅ FIX 2: Temperatures spread [0.01, 0.10, 0.30]")
    print(" ✅ FIX 3: Answer length limit 6 words (was 4)")
    print(" ✅ Generates SHORT, CRISP answers (1-3 words)")
    print(" ✅ Uses CORRECT SAQ evaluation method")
    print(" ✅ Outputs TSV + detailed results with confidence")
    print(" ✅ NEW: Creates analysis_predictions.csv for debugging")
    print(" ✅ Expected improvement: +5-10% over baseline")
    
    if not Path("saq/results/lora_model").exists():
        print("\n❌ Trained model not found at saq/results/lora_model")
        exit(1)
    
    print("\n✓ Found trained model")
    
    print("\nLoading data...")
    train_df = pd.read_csv("saq/data/train_dataset_saq.csv")
    test_df = pd.read_csv("saq/data/test_dataset_saq.csv")
    
    val_split = 0.2
    split_idx = int(len(train_df) * (1 - val_split))
    val_data = train_df.iloc[split_idx:].to_dict('records')
    
    test_data = test_df.to_dict('records')
    for i, record in enumerate(test_data):
        if 'ID' in record:
            record['id'] = record['ID']
    
    print(f"✓ Loaded {len(val_data)} validation samples")
    print(f"✓ Loaded {len(test_data)} test samples")
    
    evaluator = ModelEvaluatorWithEnsemble(
        model_path="saq/results/lora_model",
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token=HF_TOKEN,
    )
    
    # Evaluate validation
    eval_results = evaluator.evaluate_on_validation(val_data, use_ensemble=True)
    
    # Create analysis CSV (NEW)
    analysis_df = evaluator.create_analysis_predictions(
        val_data,
        output_path="saq/results/analysis_predictions.csv"
    )
    
    # Predict test
    test_predictions = evaluator.predict_on_test(
        test_data,
        output_path="saq/results/saq_prediction.tsv",
        use_ensemble=True
    )
    
    print("\n" + "="*70)
    print("✅ COMPLETED")
    print("="*70)
    print(f"\nValidation Accuracy: {eval_results.get('accuracy', 0):.2%}")
    print(f"\nOutput files:")
    print(f" • saq/results/saq_prediction.tsv (main submission file)")
    print(f" • saq/results/saq_prediction_detailed.tsv (with confidence scores)")
    print(f" • saq/results/analysis_predictions.csv (NEW: for debugging)")
    print(f"\nAnalysis CSV contains:")
    print(f" • prediction_1, prediction_2, prediction_3 (all 3 ensemble predictions)")
    print(f" • final_prediction (voted winner)")
    print(f" • confidence (vote score)")
    print(f" • is_correct (YES/NO)")
    print(f" • correct_answer (ground truth)")
    print(f"\nUse analysis_predictions.csv to:")
    print(f" • See what model predicts vs correct answers")
    print(f" • Identify patterns in errors")
    print(f" • Verify sampling creates variation")
    print(f" • Decide if Phase 2 model fix is needed")
    print(f"\nEnsemble settings (WITH FIXES):")
    print(f" • Temperatures: {ENSEMBLE_CONFIG['temperatures']}")
    print(f" • Sampling: Enabled (do_sample=True)")
    print(f" • Max answer words: 6")
    print(f" • Voting method: {ENSEMBLE_CONFIG['voting_method']}")
    print(f"\nExpected improvement: +5-10% over baseline 58%")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()