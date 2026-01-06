import re
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AnswerNormalizer:
    """Normalize answers for flexible comparison."""
    
    # Synonym mappings
    SYNONYMS = {
        'soccer': 'football',
        'football': 'football',
        'american football': 'football',
        'gridiron': 'football',
        'tea': 'tea',
        'chai': 'tea',
        'kung fu': 'martial arts',
        'martial arts': 'martial arts',
        'karate': 'martial arts',
        'taekwondo': 'martial arts',
        'uk': 'united kingdom',
        'us': 'united states',
        'usa': 'united states',
        'america': 'united states',
        'cn': 'china',
        'prc': 'china',
        'ir': 'iran',
    }
    
    @classmethod
    def normalize(cls, answer: str) -> str:
        """
        Normalize answer: lowercase, strip, apply synonyms.
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer
        """
        if not isinstance(answer, str):
            return str(answer).lower().strip()
        
        normalized = answer.lower().strip()
        return cls.SYNONYMS.get(normalized, normalized)
    
    @classmethod
    def add_synonym(cls, source: str, target: str):
        """Add custom synonym mapping."""
        cls.SYNONYMS[source.lower()] = target.lower()


class FormatValidator:
    """Validate and extract formatted answers."""
    
    @staticmethod
    def extract_time_hhmm(answer: str) -> Optional[str]:
        """
        Extract time in HHMM format (e.g., 1800, 0900).
        
        Args:
            answer: Raw answer
            
        Returns:
            HHMM string or None if not found
        """
        # Look for 4-digit time pattern
        match = re.search(r'\b([0-9]|2[0-3])([0-5][0-9])\b', str(answer))
        if match:
            return match.group(1) + match.group(2)
        
        # Try colon format and convert
        match = re.search(r'\b(?[0-9]|2[0-3]):([0-5][0-9])\b', str(answer))
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"{hour:02d}{minute:02d}"
        
        return None
    
    @staticmethod
    def extract_date_mmdd(answer: str) -> Optional[str]:
        """
        Extract date in MMDD format (e.g., 1225, 0101).
        
        Args:
            answer: Raw answer
            
        Returns:
            MMDD string or None if not found
        """
        # Look for MMDD pattern (01-12, 01-31)
        match = re.search(r'\b(0[1-9]|1[0-2])(0[1-9]|[0-9]|3)\b', str(answer))
        if match:
            return match.group(1) + match.group(2)
        
        # Try slash format and convert
        match = re.search(r'\b(0?[1-9]|1[0-2])/(0?[1-9]|?[0-9]|3)\b', str(answer))
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            return f"{month:02d}{day:02d}"
        
        return None
    
    @staticmethod
    def extract_number(answer: str) -> Optional[str]:
        """
        Extract first numeric value from answer.
        
        Args:
            answer: Raw answer
            
        Returns:
            Number as string or None if not found
        """
        match = re.search(r'\b(\d+)\b', str(answer))
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def validate_hhmm(answer: str) -> bool:
        """Check if answer is valid HHMM format."""
        return FormatValidator.extract_time_hhmm(answer) is not None
    
    @staticmethod
    def validate_mmdd(answer: str) -> bool:
        """Check if answer is valid MMDD format."""
        return FormatValidator.extract_date_mmdd(answer) is not None


class Evaluator:
    """
    Comprehensive evaluation framework for answer matching.
    Handles:
    - Exact matching
    - Format validation
    - Multiple acceptable answers
    - Detailed error analysis
    """
    
    def __init__(self, normalizer: Optional[AnswerNormalizer] = None):
        """
        Initialize evaluator.
        
        Args:
            normalizer: Custom AnswerNormalizer instance
        """
        self.normalizer = normalizer or AnswerNormalizer()
        self.evaluation_log = []
    
    def evaluate(
        self,
        predicted_answer: str,
        acceptable_answers: List[str],
        question: str,
        question_id: str = "",
        format_type: str = "TEXT"
    ) -> Tuple[bool, Dict]:
        """
        Evaluate if prediction matches any acceptable answer.
        
        Args:
            predicted_answer: Model's predicted answer
            acceptable_answers: List of correct answers
            question: Question text (for context)
            question_id: Question ID (for logging)
            format_type: Expected format (TEXT, HHMM, MMDD, NUMERIC)
            
        Returns:
            Tuple of (is_correct: bool, details: dict)
        """
        details = {
            'question_id': question_id,
            'predicted': predicted_answer,
            'acceptable': acceptable_answers,
            'format': format_type,
            'correct': False,
            'match_type': None,
            'error': None
        }
        
        try:
            # Step 1: Format-specific matching
            if format_type == 'HHMM':
                result, match_type = self._evaluate_hhmm(predicted_answer, acceptable_answers)
            elif format_type == 'MMDD':
                result, match_type = self._evaluate_mmdd(predicted_answer, acceptable_answers)
            elif format_type == 'NUMERIC':
                result, match_type = self._evaluate_numeric(predicted_answer, acceptable_answers)
            else:  # TEXT
                result, match_type = self._evaluate_text(predicted_answer, acceptable_answers)
            
            details['correct'] = result
            details['match_type'] = match_type
            
        except Exception as e:
            details['error'] = str(e)
            logger.warning(f"Evaluation error for {question_id}: {e}")
        
        self.evaluation_log.append(details)
        return details['correct'], details
    
    def _evaluate_text(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate text answer."""
        pred_norm = self.normalizer.normalize(predicted)
        
        for acc in acceptable:
            acc_norm = self.normalizer.normalize(acc)
            
            # Exact match
            if pred_norm == acc_norm:
                return True, 'exact_match'
            
            # Substring match (for multi-word answers)
            if len(pred_norm) > 3 and len(acc_norm) > 3:
                if pred_norm in acc_norm or acc_norm in pred_norm:
                    return True, 'substring_match'
        
        return False, 'no_match'
    
    def _evaluate_hhmm(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate HHMM format answer."""
        extracted_pred = FormatValidator.extract_time_hhmm(predicted)
        if not extracted_pred:
            return False, 'format_error'
        
        for acc in acceptable:
            extracted_acc = FormatValidator.extract_time_hhmm(acc)
            if extracted_acc and extracted_pred == extracted_acc:
                return True, 'format_match'
        
        return False, 'no_match'
    
    def _evaluate_mmdd(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate MMDD format answer."""
        extracted_pred = FormatValidator.extract_date_mmdd(predicted)
        if not extracted_pred:
            return False, 'format_error'
        
        for acc in acceptable:
            extracted_acc = FormatValidator.extract_date_mmdd(acc)
            if extracted_acc and extracted_pred == extracted_acc:
                return True, 'format_match'
        
        return False, 'no_match'
    
    def _evaluate_numeric(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate numeric format answer."""
        extracted_pred = FormatValidator.extract_number(predicted)
        if not extracted_pred:
            return False, 'format_error'
        
        acceptable_nums = []
        for acc in acceptable:
            extracted = FormatValidator.extract_number(acc)
            if extracted:
                acceptable_nums.append(extracted)
        
        if extracted_pred in acceptable_nums:
            return True, 'numeric_match'
        
        return False, 'no_match'
    
    def evaluate_batch(
        self,
        predictions: List[str],
        acceptable_answers_list: List[List[str]],
        questions: List[str],
        question_ids: List[str],
        format_types: List[str]
    ) -> Tuple[float, List[Dict], Dict]:
        """
        Evaluate multiple predictions.
        
        Args:
            predictions: List of predicted answers
            acceptable_answers_list: List of lists of acceptable answers
            questions: List of questions
            question_ids: List of question IDs
            format_types: List of format types
            
        Returns:
            Tuple of (accuracy: float, details_list: List, summary: Dict)
        """
        assert len(predictions) == len(acceptable_answers_list)
        
        results = []
        correct_count = 0
        match_type_counts = {}
        format_error_count = 0
        
        for pred, acceptable, question, q_id, fmt in zip(
            predictions, acceptable_answers_list, questions, question_ids, format_types
        ):
            is_correct, details = self.evaluate(
                pred, acceptable, question, q_id, fmt
            )
            results.append(details)
            
            if is_correct:
                correct_count += 1
            
            match_type = details['match_type']
            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1
            
            if details['error']:
                format_error_count += 1
        
        accuracy = correct_count / len(predictions) if predictions else 0
        
        summary = {
            'total': len(predictions),
            'correct': correct_count,
            'accuracy': accuracy,
            'match_types': match_type_counts,
            'format_errors': format_error_count
        }
        
        return accuracy, results, summary
    
    def evaluate_by_country(
        self,
        predictions: List[str],
        acceptable_answers_list: List[List[str]],
        questions: List[str],
        question_ids: List[str],
        format_types: List[str],
        countries: List[str]
    ) -> Dict[str, Dict]:
        """
        Evaluate predictions grouped by country.
        
        Returns:
            Dictionary mapping country code to accuracy metrics
        """
        accuracy, results, summary = self.evaluate_batch(
            predictions, acceptable_answers_list, questions, question_ids, format_types
        )
        
        country_results = {}
        
        for country in set(countries):
            indices = [i for i, c in enumerate(countries) if c == country]
            
            country_preds = [predictions[i] for i in indices]
            country_acceptable = [acceptable_answers_list[i] for i in indices]
            country_questions = [questions[i] for i in indices]
            country_ids = [question_ids[i] for i in indices]
            country_formats = [format_types[i] for i in indices]
            
            acc, details, summary = self.evaluate_batch(
                country_preds, country_acceptable, country_questions, country_ids, country_formats
            )
            
            country_results[country] = {
                'accuracy': acc,
                'total': len(country_preds),
                'correct': summary['correct'],
                'details': details
            }
        
        return country_results
    
    def evaluate_by_format(
        self,
        predictions: List[str],
        acceptable_answers_list: List[List[str]],
        questions: List[str],
        question_ids: List[str],
        format_types: List[str]
    ) -> Dict[str, Dict]:
        """
        Evaluate predictions grouped by format.
        
        Returns:
            Dictionary mapping format to accuracy metrics
        """
        accuracy, results, summary = self.evaluate_batch(
            predictions, acceptable_answers_list, questions, question_ids, format_types
        )
        
        format_results = {}
        
        for fmt in set(format_types):
            indices = [i for i, f in enumerate(format_types) if f == fmt]
            
            fmt_preds = [predictions[i] for i in indices]
            fmt_acceptable = [acceptable_answers_list[i] for i in indices]
            fmt_questions = [questions[i] for i in indices]
            fmt_ids = [question_ids[i] for i in indices]
            
            acc, details, summary = self.evaluate_batch(
                fmt_preds, fmt_acceptable, fmt_questions, fmt_ids,
                [fmt] * len(indices)
            )
            
            format_results[fmt] = {
                'accuracy': acc,
                'total': len(fmt_preds),
                'correct': summary['correct'],
                'details': details
            }
        
        return format_results
    
    def print_evaluation_summary(self, results: Dict):
        """Pretty print evaluation results."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        if 'accuracy' in results:
            # Single evaluation result
            print(f"Accuracy: {results['accuracy']:.1%}")
            print(f"Correct: {results['correct']}/{results['total']}")
            print(f"\nMatch Types:")
            for match_type, count in results['match_types'].items():
                pct = (count / results['total']) * 100 if results['total'] > 0 else 0
                print(f"  {match_type}: {count:4d} ({pct:5.1f}%)")
            print(f"\nFormat Errors: {results['format_errors']}")
        else:
            # Country/Format grouped results
            for key, metrics in results.items():
                print(f"\n{key}:")
                print(f"  Accuracy: {metrics['accuracy']:.1%}")
                print(f"  Correct: {metrics['correct']}/{metrics['total']}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    evaluator = Evaluator()
    
    # Test exact match
    is_correct, details = evaluator.evaluate(
        predicted_answer="Football",
        acceptable_answers=["football", "soccer", "american football"],
        question="What is the most popular sport?",
        question_id="test-001",
        format_type="TEXT"
    )
    print(f"Result: {is_correct}")
    print(f"Details: {json.dumps(details, indent=2)}")
    
    # Test time format
    is_correct, details = evaluator.evaluate(
        predicted_answer="Dinner is at 18:00",
        acceptable_answers=["1800", "2000"],
        question="What time do people eat dinner? (HHMM format)",
        question_id="test-002",
        format_type="HHMM"
    )
    print(f"\nResult: {is_correct}")
    print(f"Details: {json.dumps(details, indent=2)}")