import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
from datetime import datetime

from data_handler import DataHandler
from evaluator import Evaluator, AnswerNormalizer
from baseline_zeroshot import ZeroShotGenerator, GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Complete pipeline for data loading, generation, and evaluation.
    """
    
    def __init__(
        self,
        train_path: str,
        test_path: str,
        output_dir: str = "/results"
    ):
        """
        Initialize pipeline.
        
        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
            output_dir: Directory to save results
        """
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.llama_config = {
            "use_8bit": True,
            "use_flash_attention": True,
            "device": "auto"
        }
        
        # Initialize components
        self.data_handler = DataHandler(train_path, test_path)
        self.evaluator = Evaluator()
        self.generator = None  # Will be initialized later
        
        # Results storage
        self.train_split = None
        self.val_split = None
        self.results = {}
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    def prepare_data(self, val_size: float = 0.2):
        """
        Prepare data: split into train/val.
        
        Args:
            val_size: Validation split size
        """
        logger.info("Preparing data...")
        self.train_split, self.val_split = self.data_handler.split_train_val(
            val_size=val_size
        )
        logger.info("Data preparation complete")
    
    def create_evaluation_dataset(
        self,
        split: str = "val"
    ) -> List[Dict]:
        """
        Create dataset for evaluation with all required fields.
        
        Args:
            split: Which split to use ("val" or "test")
            
        Returns:
            List of evaluation samples
        """
        if split == "val":
            df = self.val_split
        else:
            df = self.data_handler.test_df
        
        eval_data = []
        
        for idx in df.index:
            row = df.loc[idx]
            sample = {
                'ID': row['ID'],
                'question': row['en_question'],
                'country': row['country'],
                'format': self.data_handler.detect_answer_format(row['en_question']),
                'acceptable_answers': (
                    self.data_handler.extract_all_acceptable_answers(row['annotations'])
                    if 'annotations' in row and pd.notna(row['annotations'])
                    else []
                )
            }
            eval_data.append(sample)
        
        return eval_data
    
    # ========================================================================
    # GENERATION
    # ========================================================================
    
    def initialize_generator(
        self,
        backend: str = "llama",
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.3,
        **kwargs
    ):
        """
        Initialize zero-shot generator.
        
        Args:
            backend: LLM backend ("Llama" or "huggingface")
            model_name: Model identifier
            temperature: Sampling temperature
        """
        generation_config = GenerationConfig(
            temperature=temperature,
            **kwargs
        )
        self.generator = ZeroShotGenerator(generation_config=generation_config)
        logger.info(f"Generator initialized: ({model_name})")
    
    def generate_answers(
        self,
        eval_data: List[Dict],
        template: str = "CONTEXT_AWARE",
        batch_size: int = 10
    ) -> List[str]:
        """
        Generate answers for all samples in evaluation dataset.
        
        Args:
            eval_data: List of evaluation samples
            template: Prompt template to use
            batch_size: Batch size for progress tracking
            
        Returns:
            List of generated answers
        """
        if not self.generator:
            raise ValueError("Generator not initialized. Call initialize_generator()")
        
        logger.info(f"Generating answers for {len(eval_data)} samples...")
        
        questions = [sample['question'] for sample in eval_data]
        countries = [sample['country'] for sample in eval_data]
        formats = [sample['format'] for sample in eval_data]
        
        answers = self.generator.generate_batch(
            questions=questions,
            template=template,
            batch_size=batch_size
        )
        
        logger.info(f"Generated {len(answers)} answers")
        return answers
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    def evaluate_predictions(
        self,
        eval_data: List[Dict],
        predictions: List[str]
    ) -> Tuple[float, Dict, Dict]:
        """
        Evaluate predictions against acceptable answers.
        
        Args:
            eval_data: List of evaluation samples
            predictions: List of predicted answers
            
        Returns:
            Tuple of (overall_accuracy, country_results, format_results)
        """
        logger.info("Evaluating predictions...")
        
        # Prepare evaluation inputs
        acceptable_answers_list = [sample['acceptable_answers'] for sample in eval_data]
        questions = [sample['question'] for sample in eval_data]
        question_ids = [sample['ID'] for sample in eval_data]
        formats = [sample['format'] for sample in eval_data]
        countries = [sample['country'] for sample in eval_data]
        
        # Overall evaluation
        overall_accuracy, _, summary = self.evaluator.evaluate_batch(
            predictions=predictions,
            acceptable_answers_list=acceptable_answers_list,
            questions=questions,
            question_ids=question_ids,
            format_types=formats
        )
        
        logger.info(f"Overall Accuracy: {overall_accuracy:.1%}")
        
        # By country
        country_results = self.evaluator.evaluate_by_country(
            predictions=predictions,
            acceptable_answers_list=acceptable_answers_list,
            questions=questions,
            question_ids=question_ids,
            format_types=formats,
            countries=countries
        )
        
        logger.info("Accuracy by country:")
        for country, metrics in country_results.items():
            logger.info(f"  {country}: {metrics['accuracy']:.1%}")
        
        # By format
        format_results = self.evaluator.evaluate_by_format(
            predictions=predictions,
            acceptable_answers_list=acceptable_answers_list,
            questions=questions,
            question_ids=question_ids,
            format_types=formats
        )
        
        logger.info("Accuracy by format:")
        for fmt, metrics in format_results.items():
            logger.info(f"  {fmt}: {metrics['accuracy']:.1%}")
        
        return overall_accuracy, country_results, format_results
    
    # ========================================================================
    # RESULTS SAVING
    # ========================================================================
    
    def save_results(
        self,
        eval_data: List[Dict],
        predictions: List[str],
        overall_accuracy: float,
        country_results: Dict,
        format_results: Dict,
        split: str = "val"
    ):
        """
        Save evaluation results to files.
        
        Args:
            eval_data: Evaluation samples
            predictions: Predicted answers
            overall_accuracy: Overall accuracy
            country_results: Results by country
            format_results: Results by format
            split: Which split being evaluated
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_subdir = self.output_dir / f"results_{split}_{timestamp}"
        results_subdir.mkdir(exist_ok=True, parents=True)
        
        # 1. Save predictions CSV
        predictions_df = pd.DataFrame({
            'ID': [s['ID'] for s in eval_data],
            'question': [s['question'] for s in eval_data],
            'country': [s['country'] for s in eval_data],
            'predicted_answer': predictions,
            'acceptable_answers': [';'.join(s['acceptable_answers']) for s in eval_data],
            'format': [s['format'] for s in eval_data]
        })
        
        predictions_path = results_subdir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
        
        # 2. Save evaluation metrics JSON
        metrics = {
            'split': split,
            'timestamp': timestamp,
            'overall_accuracy': overall_accuracy,
            'country_results': {
                country: {
                    'accuracy': metrics['accuracy'],
                    'correct': metrics['correct'],
                    'total': metrics['total']
                }
                for country, metrics in country_results.items()
            },
            'format_results': {
                fmt: {
                    'accuracy': metrics['accuracy'],
                    'correct': metrics['correct'],
                    'total': metrics['total']
                }
                for fmt, metrics in format_results.items()
            }
        }
        
        metrics_path = results_subdir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # 3. Save detailed evaluation log
        eval_log_path = results_subdir / "evaluation_log.json"
        with open(eval_log_path, 'w') as f:
            json.dump(self.evaluator.evaluation_log, f, indent=2)
        logger.info(f"Saved evaluation log to {eval_log_path}")
        
        return results_subdir
    
    # ========================================================================
    # FULL PIPELINE EXECUTION
    # ========================================================================
    
    def run_full_evaluation(
        self,
        backend: str = "llama",
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        template: str = "CONTEXT_AWARE",
        val_size: float = 0.2,
        split: str = "val"
    ) -> Dict:
        """
        Run complete pipeline: prepare data, generate, evaluate, save.
        
        Args:
            backend: LLM backend
            model_name: Model name
            template: Prompt template
            val_size: Validation split size
            split: Evaluation split ("val" or "test")
            
        Returns:
            Dictionary with all results
        """
        logger.info("Starting full evaluation pipeline...")
        
        # Step 1: Prepare data
        self.prepare_data(val_size=val_size)
        
        # Step 2: Initialize generator
        self.initialize_generator(backend=backend, model_name=model_name)
        
        # Step 3: Create evaluation dataset
        eval_data = self.create_evaluation_dataset(split=split)
        logger.info(f"Created evaluation dataset: {len(eval_data)} samples")
        
        # Step 4: Generate answers
        predictions = self.generate_answers(eval_data, template=template)
        
        # Step 5: Evaluate
        overall_acc, country_res, format_res = self.evaluate_predictions(
            eval_data, predictions
        )
        
        # Step 6: Save results
        results_dir = self.save_results(
            eval_data, predictions, overall_acc, country_res, format_res, split
        )
        
        # Return summary
        summary = {
            'overall_accuracy': overall_acc,
            'country_accuracies': {c: m['accuracy'] for c, m in country_res.items()},
            'format_accuracies': {f: m['accuracy'] for f, m in format_res.items()},
            'results_directory': str(results_dir)
        }
        
        logger.info(f"Evaluation complete! Results saved to {results_dir}")
        
        return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(
        train_path="saq/data/train_dataset_saq.csv",
        test_path="saq/data/test_dataset_saq.csv",
        output_dir="saq/results"
    )
    
    # Run evaluation
    # NOTE: This requires Llama API key set in environment
    results = pipeline.run_full_evaluation(
        backend="llama",
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        template="CONTEXT_AWARE",
        val_size=0.2,
        split="val"
    )
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    print("\nBy Country:")
    for country, acc in results['country_accuracies'].items():
        print(f"  {country}: {acc:.1%}")
    print("\nBy Format:")
    for fmt, acc in results['format_accuracies'].items():
        print(f"  {fmt}: {acc:.1%}")
    print(f"\nResults saved to: {results['results_directory']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
