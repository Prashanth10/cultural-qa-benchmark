import re
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FORMAT DETECTION & EXTRACTION (for improved accuracy)
# ============================================================================

def detect_format_requirement(question: str) -> str:
    """
    Detect what format the question expects
    
    Args:
        question: The question text
    
    Returns:
        Format type: "TEXT", "HHMM", "MMDD", or "NUMERIC"
    """
    
    q_lower = question.lower()
    
    # Check for time format
    if "hhmm" in q_lower or ("provide in" in q_lower and "format" in q_lower and "time" in q_lower):
        return "HHMM"
    
    # Check for date format
    if "mmdd" in q_lower or ("provide in" in q_lower and "format" in q_lower and "date" in q_lower):
        return "MMDD"
    
    # Check for numeric format
    if "arabic numerals" in q_lower or "provide" in q_lower and "numerals" in q_lower:
        return "NUMERIC"
    if any(phrase in q_lower for phrase in ["how many", "how long", "how much", "how old"]):
        return "NUMERIC"
    
    # Default to text
    return "TEXT"


def extract_format(answer: str, format_type: str) -> str:
    """
    Extract answer in required format from model output
    
    Args:
        answer: Raw output from model (e.g., "People finish at 1800")
        format_type: Expected format (TEXT, HHMM, MMDD, NUMERIC)
    
    Returns:
        Extracted answer in correct format (e.g., "1800")
    """
    
    answer = answer.strip().lower()
    
    if format_type == "HHMM":
        match = re.search(r'\b(\d{4})\b', answer)
        if match:
            return match.group(1)
        
        match = re.search(r'(\d{1,2}):?(\d{2})\s*(am|pm)?', answer)
        if match:
            hour = int(match.group(1))
            minute = match.group(2)
            
            period = match.group(3)
            if period:
                if period == 'pm' and hour != 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
            
            return f"{hour:02d}{minute:02d}"
        
        return ""
    
    elif format_type == "MMDD":
        match = re.search(r'(\d{1,2})[/-](\d{1,2})', answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{month:02d}{day:02d}"
        
        match = re.search(r'\b(\d{4})\b', answer)
        if match:
            return match.group(1)
        
        return ""
    
    elif format_type == "NUMERIC":
        match = re.search(r'(\d+(?:\.\d+)?)', answer)
        if match:
            return match.group(1)
        return ""
    
    else:  # TEXT format
        answer = re.sub(r'[^\w\s-]', ' ', answer)
        answer = ' '.join(answer.split())
        return answer[:100].strip()


def validate_format(answer: str, format_type: str) -> bool:
    """
    Validate if answer matches required format
    """
    
    answer = answer.strip()
    
    if format_type == "HHMM":
        return bool(re.match(r'^\d{4}$', answer))
    
    elif format_type == "MMDD":
        return bool(re.match(r'^\d{4}$', answer))
    
    elif format_type == "NUMERIC":
        return bool(re.match(r'^\d+(?:\.\d+)?$', answer))
    
    else:  # TEXT
        return len(answer) > 0


class AnswerNormalizer:
    """Normalize answers for flexible comparison."""

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
        """Normalize answer: lowercase, strip, apply synonyms."""
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
        """Extract time in HHMM format (e.g., 1800, 0900)."""
        match = re.search(r'\b([0-9]|2[0-3])([0-5][0-9])\b', str(answer))
        if match:
            return match.group(1) + match.group(2)

        match = re.search(r'\b([0-9]|2[0-3]):([0-5][0-9])\b', str(answer))
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"{hour:02d}{minute:02d}"

        return None

    @staticmethod
    def extract_date_mmdd(answer: str) -> Optional[str]:
        """Extract date in MMDD format (e.g., 1225, 0101)."""
        match = re.search(r'\b(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])\b', str(answer))
        if match:
            return match.group(1) + match.group(2)

        match = re.search(r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])\b', str(answer))
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            return f"{month:02d}{day:02d}"

        return None

    @staticmethod
    def extract_number(answer: str) -> Optional[str]:
        """Extract first numeric value from answer."""
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
    - Format extraction for better accuracy
    """

    def __init__(self, normalizer: Optional[AnswerNormalizer] = None):
        """Initialize evaluator."""
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
        """Evaluate if prediction matches any acceptable answer."""
        
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
            if pred_norm == acc_norm:
                return True, 'exact_match'
            if len(pred_norm) > 3 and len(acc_norm) > 3:
                if pred_norm in acc_norm or acc_norm in pred_norm:
                    return True, 'substring_match'
        return False, 'no_match'

    def _evaluate_hhmm(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate HHMM format answer with format extraction."""
        extracted_pred = extract_format(predicted, "HHMM")
        if not extracted_pred:
            return False, 'format_error'
        for acc in acceptable:
            extracted_acc = extract_format(acc, "HHMM")
            if extracted_acc and extracted_pred == extracted_acc:
                return True, 'format_match'
        return False, 'no_match'

    def _evaluate_mmdd(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate MMDD format answer with format extraction."""
        extracted_pred = extract_format(predicted, "MMDD")
        if not extracted_pred:
            return False, 'format_error'
        for acc in acceptable:
            extracted_acc = extract_format(acc, "MMDD")
            if extracted_acc and extracted_pred == extracted_acc:
                return True, 'format_match'
        return False, 'no_match'

    def _evaluate_numeric(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """Evaluate numeric format answer with format extraction."""
        extracted_pred = extract_format(predicted, "NUMERIC")
        if not extracted_pred:
            return False, 'format_error'
        acceptable_nums = []
        for acc in acceptable:
            extracted = extract_format(acc, "NUMERIC")
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
        """Evaluate multiple predictions."""
        
        assert len(predictions) == len(acceptable_answers_list)

        results = []
        correct_count = 0
        match_type_counts = {}
        format_error_count = 0

        for pred, acceptable, question, q_id, fmt in zip(
            predictions, acceptable_answers_list, questions, question_ids, format_types
        ):
            is_correct, details = self.evaluate(pred, acceptable, question, q_id, fmt)
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
        """Evaluate predictions grouped by country."""
        
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
        """Evaluate predictions grouped by format."""
        
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
            print(f"Accuracy: {results['accuracy']:.1%}")
            print(f"Correct: {results['correct']}/{results['total']}")
            print(f"\nMatch Types:")
            for match_type, count in results['match_types'].items():
                pct = (count / results['total']) * 100 if results['total'] > 0 else 0
                print(f" {match_type}: {count:4d} ({pct:5.1f}%)")
            print(f"\nFormat Errors: {results['format_errors']}")
        else:
            for key, metrics in results.items():
                print(f"\n{key}:")
                print(f" Accuracy: {metrics['accuracy']:.1%}")
                print(f" Correct: {metrics['correct']}/{metrics['total']}")
        print("="*70 + "\n")


if __name__ == "__main__":
    evaluator = Evaluator()
    
    is_correct, details = evaluator.evaluate(
        predicted_answer="Football",
        acceptable_answers=["football", "soccer"],
        question="What is the most popular sport?",
        question_id="test-001",
        format_type="TEXT"
    )
    print(f"Result: {is_correct}")
    print(f"Details: {json.dumps(details, indent=2)}")