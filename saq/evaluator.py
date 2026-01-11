import re
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# NEW: FORMAT DETECTION & EXTRACTION (3 NEW FUNCTIONS)
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
    if "arabic numerals" in q_lower or ("provide" in q_lower and "numerals" in q_lower):
        return "NUMERIC"
    if any(phrase in q_lower for phrase in ["how many", "how long", "how much", "how old"]):
        return "NUMERIC"
    
    # Default to text
    return "TEXT"


def extract_format(answer: str, format_type: str) -> str:
    """
    Extract answer in required format from model output
    
    Args:
        answer: Raw output from model (e.g., "People leave at 1800")
        format_type: Expected format (TEXT, HHMM, MMDD, NUMERIC)
    
    Returns:
        Extracted answer in correct format (e.g., "1800")
    """
    answer = answer.strip().lower()
    
    if format_type == "HHMM":
        # Try exact 4-digit pattern first (1800, 0700, etc.)
        match = re.search(r'\b(\d{4})\b', answer)
        if match:
            return match.group(1)
        
        # Try HH:MM or H:MM pattern (6:00 PM, 18:00, etc.)
        match = re.search(r'(\d{1,2}):?(\d{2})\s*(am|pm)?', answer)
        if match:
            hour = int(match.group(1))
            minute = match.group(2)
            
            # Convert to 24-hour format if needed
            period = match.group(3)
            if period:
                if period == 'pm' and hour != 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
            
            return f"{hour:02d}{minute:02d}"
        
        return ""
    
    elif format_type == "MMDD":
        # Try M/D or MM/DD format
        match = re.search(r'(\d{1,2})[/-](\d{1,2})', answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{month:02d}{day:02d}"
        
        # Try direct 4-digit pattern
        match = re.search(r'\b(\d{4})\b', answer)
        if match:
            return match.group(1)
        
        return ""
    
    elif format_type == "NUMERIC":
        # Extract first number (integer or decimal)
        match = re.search(r'(\d+(?:\.\d+)?)', answer)
        if match:
            return match.group(1)
        return ""
    
    else:  # TEXT format
        # Remove punctuation, clean up whitespace
        answer = re.sub(r'[^\w\s-]', ' ', answer)
        answer = ' '.join(answer.split())
        return answer[:100].strip()


def validate_format(answer: str, format_type: str) -> bool:
    """
    Validate if answer matches required format
    
    Args:
        answer: The answer to validate
        format_type: Expected format
    
    Returns:
        True if format is valid, False otherwise
    """
    answer = answer.strip()
    
    if format_type == "HHMM":
        # Must be exactly 4 digits
        return bool(re.match(r'^\d{4}$', answer))
    
    elif format_type == "MMDD":
        # Must be exactly 4 digits
        return bool(re.match(r'^\d{4}$', answer))
    
    elif format_type == "NUMERIC":
        # Must be number (int or decimal)
        return bool(re.match(r'^\d+(?:\.\d+)?$', answer))
    
    else:  # TEXT
        # Must have at least 1 character
        return len(answer) > 0


# ============================================================================
# EXISTING CODE: Keep your existing AnswerNormalizer, FormatValidator classes
# ============================================================================
# [KEEP YOUR EXISTING CLASSES HERE - Don't delete them]
# Your AnswerNormalizer class
# Your FormatValidator class
# etc.


# ============================================================================
# EVALUATOR CLASS (UPDATED with NEW METHODS)
# ============================================================================

class Evaluator:
    """
    Main evaluator class for answer evaluation
    """
    
    def __init__(self):
        """Initialize evaluator"""
        # Add any initialization you need
        pass
    
    # ========================================================================
    # UPDATED METHOD #1: _evaluate_hhmm (WITH FORMAT EXTRACTION)
    # ========================================================================
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
    
    # ========================================================================
    # UPDATED METHOD #2: _evaluate_mmdd (WITH FORMAT EXTRACTION)
    # ========================================================================
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
    
    # ========================================================================
    # UPDATED METHOD #3: _evaluate_numeric (WITH FORMAT EXTRACTION)
    # ========================================================================
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
    
    # ========================================================================
    # KEEP YOUR EXISTING METHODS (evaluate_batch, etc.)
    # ========================================================================
    # [KEEP ALL YOUR OTHER EXISTING METHODS HERE]
    # Just replace the 3 methods above with the new versions
    
    def evaluate_batch(
        self,
        predictions: List[str],
        acceptable_answers_list: List[List[str]],
        questions: List[str],
        question_ids: List[str],
        format_types: List[str] = None
    ) -> Tuple[float, Dict, Dict]:
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: List of model predictions
            acceptable_answers_list: List of acceptable answers for each question
            questions: List of questions
            question_ids: List of question IDs
            format_types: List of format types (TEXT, HHMM, MMDD, NUMERIC)
        
        Returns:
            Tuple of (accuracy, results_dict, summary_dict)
        """
        if format_types is None:
            format_types = ['TEXT'] * len(predictions)
        
        correct = 0
        total = len(predictions)
        results = []
        match_types = {}
        format_errors = 0
        
        for pred, acceptable, q, qid, fmt_type in zip(
            predictions, acceptable_answers_list, questions, question_ids, format_types
        ):
            # Evaluate based on format type
            if fmt_type == "HHMM":
                is_correct, match_type = self._evaluate_hhmm(pred, acceptable)
            elif fmt_type == "MMDD":
                is_correct, match_type = self._evaluate_mmdd(pred, acceptable)
            elif fmt_type == "NUMERIC":
                is_correct, match_type = self._evaluate_numeric(pred, acceptable)
            else:  # TEXT
                # Your existing TEXT evaluation logic
                is_correct, match_type = self._evaluate_text(pred, acceptable)
            
            # Count format errors
            if match_type == 'format_error':
                format_errors += 1
            
            # Count correct
            if is_correct:
                correct += 1
            
            # Track match types
            match_types[match_type] = match_types.get(match_type, 0) + 1
            
            # Store result
            results.append({
                'question_id': qid,
                'prediction': pred,
                'acceptable': acceptable,
                'correct': is_correct,
                'match_type': match_type,
                'format_type': fmt_type
            })
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Create summary
        summary = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'match_types': match_types,
            'format_errors': format_errors
        }
        
        return accuracy, results, summary
    
    def _evaluate_text(self, predicted: str, acceptable: List[str]) -> Tuple[bool, str]:
        """
        Evaluate TEXT format (your existing logic)
        Keep your existing TEXT evaluation method
        """
        # Your existing TEXT evaluation logic
        predicted = predicted.strip().lower()
        for acc in acceptable:
            if predicted == acc.strip().lower():
                return True, 'exact_match'
        return False, 'no_match'
