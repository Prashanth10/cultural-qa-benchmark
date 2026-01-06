import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Comprehensive data handling for SAQ (Situational Aware Questions) dataset.
    Handles:
    - Loading train and test CSV files
    - Parsing complex JSON annotations
    - Format detection and validation
    - Data splitting (train/val/test)
    - Statistics and analysis
    """
    
    # Constants
    VALID_COUNTRIES = {'US', 'GB', 'CN', 'IR'}
    FORMAT_INDICATORS = {
        'HHMM': r'(HH:MM|HHMM|time)',
        'MMDD': r'(MM/DD|MMDD|date)',
        'NUMERIC': r'(Arabic numerals|Provide.*numerals)',
        'ORDINAL': r'(e\.g\.,\s*\d+)'
    }
    
    def __init__(self, train_path: str, test_path: str):
        """
        Initialize DataHandler with file paths.
        
        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
        """
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        
        self.train_df = None
        self.test_df = None
        self.train_split = None  # For train/val split
        self.val_split = None
        
        self._load_data()
    
    def _load_data(self):
        """Load CSV files with proper encoding handling."""
        try:
            logger.info(f"Loading training data from {self.train_path}")
            self.train_df = pd.read_csv(self.train_path, encoding='utf-8')
            
            logger.info(f"Loading test data from {self.test_path}")
            self.test_df = pd.read_csv(self.test_path, encoding='utf-8')
            
            logger.info(f"Training samples: {len(self.train_df)}")
            logger.info(f"Test samples: {len(self.test_df)}")
            
            # Validate columns
            self._validate_columns()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_columns(self):
        """Validate that required columns exist."""
        required_train_cols = {'ID', 'question', 'en_question', 'annotations', 'idks', 'country'}
        required_test_cols = {'ID', 'question', 'en_question', 'country'}
        
        train_cols = set(self.train_df.columns)
        test_cols = set(self.test_df.columns)
        
        missing_train = required_train_cols - train_cols
        missing_test = required_test_cols - test_cols
        
        if missing_train:
            raise ValueError(f"Missing columns in training data: {missing_train}")
        if missing_test:
            raise ValueError(f"Missing columns in test data: {missing_test}")
        
        logger.info("✓ Column validation passed")
    
    # ============================================================================
    # ANNOTATION PARSING
    # ============================================================================
    
    def parse_annotations(self, annotation_str: str) -> List[Dict]:
        """
        Parse annotation JSON string to dictionary.
        
        Handles various JSON string formats with single/double quotes.
        
        Args:
            annotation_str: String representation of annotations list
            
        Returns:
            List of annotation dictionaries with keys: 'answers', 'en_answers', 'count'
        """
        if not isinstance(annotation_str, str) or annotation_str.strip() == '':
            return []
        
        try:
            # Convert single quotes to double quotes for valid JSON
            fixed_str = annotation_str.replace("'", '"')
            annotations = json.loads(fixed_str)
            return annotations if isinstance(annotations, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse annotation: {annotation_str[:100]}")
            return []
    
    def extract_all_acceptable_answers(self, annotation_str: str) -> List[str]:
        """
        Extract all unique acceptable answers in English from annotations.
        
        Args:
            annotation_str: Annotation JSON string
            
        Returns:
            List of normalized acceptable answers
        """
        annotations = self.parse_annotations(annotation_str)
        answers = set()
        
        for item in annotations:
            if isinstance(item, dict):
                en_answers = item.get('en_answers', [])
                if isinstance(en_answers, list):
                    for answer in en_answers:
                        # Normalize: lowercase, strip whitespace
                        normalized = str(answer).lower().strip()
                        if normalized:
                            answers.add(normalized)
        
        return sorted(list(answers))
    
    def get_answer_distribution(self, annotation_str: str) -> Dict[str, int]:
        """
        Get answer frequency distribution.
        
        Args:
            annotation_str: Annotation JSON string
            
        Returns:
            Dictionary mapping answers to counts
        """
        annotations = self.parse_annotations(annotation_str)
        distribution = {}
        
        for item in annotations:
            if isinstance(item, dict):
                en_answers = item.get('en_answers', [])
                count = item.get('count', 0)
                
                if isinstance(en_answers, list):
                    for answer in en_answers:
                        normalized = str(answer).lower().strip()
                        distribution[normalized] = count
        
        return distribution
    
    def parse_idks(self, idks_str: str) -> Dict[str, int]:
        """
        Parse IDK (I Don't Know) statistics.
        
        Args:
            idks_str: String representation of IDK dict
            
        Returns:
            Dictionary with keys: 'idk', 'no-answer', 'not-applicable'
        """
        if not isinstance(idks_str, str) or idks_str.strip() == '':
            return {'idk': 0, 'no-answer': 0, 'not-applicable': 0}
        
        try:
            fixed_str = idks_str.replace("'", '"')
            idks = json.loads(fixed_str)
            return idks if isinstance(idks, dict) else {}
        except json.JSONDecodeError:
            return {}
    
    # ============================================================================
    # FORMAT DETECTION
    # ============================================================================
    
    def detect_answer_format(self, question: str) -> str:
        """
        Detect required answer format from question text.
        
        Args:
            question: Question string
            
        Returns:
            Format type: 'HHMM', 'MMDD', 'NUMERIC', 'TEXT'
        """
        question_upper = question.upper()
        
        # Time format (HH:MM or HHMM)
        if re.search(self.FORMAT_INDICATORS['HHMM'], question_upper):
            return 'HHMM'
        
        # Date format (MM/DD or MMDD)
        if re.search(self.FORMAT_INDICATORS['MMDD'], question_upper):
            return 'MMDD'
        
        # Numeric format
        if re.search(self.FORMAT_INDICATORS['NUMERIC'], question_upper):
            return 'NUMERIC'
        
        # Default to TEXT
        return 'TEXT'
    
    def detect_all_formats(self) -> Dict[str, int]:
        """
        Analyze format distribution in training data.
        
        Returns:
            Dictionary with format counts
        """
        format_counts = {'HHMM': 0, 'MMDD': 0, 'NUMERIC': 0, 'TEXT': 0}
        
        for question in self.train_df['en_question']:
            fmt = self.detect_answer_format(question)
            format_counts[fmt] += 1
        
        logger.info("Format distribution:")
        for fmt, count in format_counts.items():
            pct = (count / len(self.train_df)) * 100
            logger.info(f"  {fmt}: {count} ({pct:.1f}%)")
        
        return format_counts
    
    # ============================================================================
    # DATA SPLITTING
    # ============================================================================
    
    def split_train_val(self, val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets.
        
        Stratified by country to maintain distribution.
        
        Args:
            val_size: Fraction for validation (default 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info(f"Splitting training data: {100-val_size*100:.0f}% train, {val_size*100:.0f}% val")
        
        # Use stratified split by country
        self.train_split, self.val_split = train_test_split(
            self.train_df,
            test_size=val_size,
            stratify=self.train_df['country'],
            random_state=random_state
        )
        
        logger.info(f"Train set: {len(self.train_split)} samples")
        logger.info(f"Val set: {len(self.val_split)} samples")
        
        # Log country distribution
        logger.info("Train set country distribution:")
        for country, count in self.train_split['country'].value_counts().items():
            logger.info(f"  {country}: {count}")
        
        logger.info("Val set country distribution:")
        for country, count in self.val_split['country'].value_counts().items():
            logger.info(f"  {country}: {count}")
        
        return self.train_split, self.val_split
    
    # ============================================================================
    # DATA STATISTICS & ANALYSIS
    # ============================================================================
    
    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive statistics about dataset.
        
        Args:
            df: DataFrame to analyze (uses train_df if None)
            
        Returns:
            Dictionary with various statistics
        """
        if df is None:
            df = self.train_df
        
        stats = {}
        
        # Basic counts
        stats['total_samples'] = len(df)
        stats['unique_countries'] = df['country'].nunique()
        
        # Country distribution
        stats['country_distribution'] = df['country'].value_counts().to_dict()
        
        # Answer statistics (only for training data with annotations)
        if 'annotations' in df.columns:
            answer_counts = []
            uncertainty_counts = {'idk': 0, 'no-answer': 0, 'not-applicable': 0}
            
            for _, row in df.iterrows():
                acceptable = self.extract_all_acceptable_answers(row['annotations'])
                answer_counts.append(len(acceptable))
                
                idks = self.parse_idks(row['idks'])
                for key in uncertainty_counts:
                    uncertainty_counts[key] += idks.get(key, 0)
            
            stats['avg_answers_per_question'] = np.mean(answer_counts) if answer_counts else 0
            stats['min_answers'] = min(answer_counts) if answer_counts else 0
            stats['max_answers'] = max(answer_counts) if answer_counts else 0
            stats['uncertainty_stats'] = uncertainty_counts
        
        # Format distribution
        stats['format_distribution'] = {}
        for question in df['en_question']:
            fmt = self.detect_answer_format(question)
            stats['format_distribution'][fmt] = stats['format_distribution'].get(fmt, 0) + 1
        
        return stats
    
    def print_statistics(self, df: Optional[pd.DataFrame] = None):
        """Pretty print dataset statistics."""
        if df is None:
            df = self.train_df
        
        stats = self.get_statistics(df)
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Unique countries: {stats['unique_countries']}")
        
        print("\nCountry Distribution:")
        for country, count in stats['country_distribution'].items():
            pct = (count / stats['total_samples']) * 100
            print(f"  {country}: {count:4d} ({pct:5.1f}%)")
        
        if 'avg_answers_per_question' in stats:
            print(f"\nAnswer Statistics:")
            print(f"  Avg answers per question: {stats['avg_answers_per_question']:.2f}")
            print(f"  Min answers: {stats['min_answers']}")
            print(f"  Max answers: {stats['max_answers']}")
            
            print(f"\nUncertainty Statistics:")
            for key, count in stats['uncertainty_stats'].items():
                print(f"  {key}: {count}")
        
        print(f"\nFormat Distribution:")
        for fmt, count in stats['format_distribution'].items():
            pct = (count / stats['total_samples']) * 100
            print(f"  {fmt}: {count:4d} ({pct:5.1f}%)")
        print("="*60 + "\n")
    
    # ============================================================================
    # DATA ACCESS METHODS
    # ============================================================================
    
    def get_sample(self, idx: int, use_split: str = 'train') -> Dict:
        """
        Get a single sample with all parsed information.
        
        Args:
            idx: Row index
            use_split: Which split to use ('train', 'val', 'test')
            
        Returns:
            Dictionary with parsed sample data
        """
        if use_split == 'train':
            df = self.train_split if self.train_split is not None else self.train_df
        elif use_split == 'val':
            df = self.val_split
        else:  # test
            df = self.test_df
        
        row = df.iloc[idx]
        
        sample = {
            'ID': row['ID'],
            'question_native': row['question'],
            'question_en': row['en_question'],
            'country': row['country'],
            'answer_format': self.detect_answer_format(row['en_question'])
        }
        
        # Add annotations if available
        if 'annotations' in row and pd.notna(row['annotations']):
            sample['acceptable_answers'] = self.extract_all_acceptable_answers(row['annotations'])
            sample['answer_distribution'] = self.get_answer_distribution(row['annotations'])
            sample['idks'] = self.parse_idks(row['idks'])
        
        return sample
    
    def get_samples_by_country(self, country: str, use_split: str = 'train') -> List[Dict]:
        """Get all samples for a specific country."""
        if use_split == 'train':
            df = self.train_split if self.train_split is not None else self.train_df
        elif use_split == 'val':
            df = self.val_split
        else:
            df = self.test_df
        
        samples = []
        country_df = df[df['country'] == country]
        
        for idx in country_df.index:
            samples.append(self.get_sample(idx, use_split))
        
        return samples
    
    def get_samples_by_format(self, fmt: str, use_split: str = 'train') -> List[Dict]:
        """Get all samples with specific format requirement."""
        if use_split == 'train':
            df = self.train_split if self.train_split is not None else self.train_df
        elif use_split == 'val':
            df = self.val_split
        else:
            df = self.test_df
        
        samples = []
        for idx in df.index:
            sample = self.get_sample(idx, use_split)
            if sample['answer_format'] == fmt:
                samples.append(sample)
        
        return samples
    
    def export_for_evaluation(self, use_split: str = 'val') -> List[Dict]:
        """
        Export dataset in format suitable for evaluation.
        
        Returns list of dicts with: ID, question, country, format, acceptable_answers
        """
        if use_split == 'train':
            df = self.train_split if self.train_split is not None else self.train_df
        elif use_split == 'val':
            df = self.val_split
        else:
            df = self.test_df
        
        evaluation_data = []
        for idx in df.index:
            sample = self.get_sample(idx, use_split)
            evaluation_data.append(sample)
        
        return evaluation_data


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize data handler
    handler = DataHandler(
        train_path='saq/data/train_dataset_saq.csv',
        test_path='saq/data/test_dataset_saq.csv'
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    handler.print_statistics(handler.train_df)
    
    print("\n" + "="*60)
    print("TEST DATA STATISTICS")
    print("="*60)
    handler.print_statistics(handler.test_df)
    
    # Split training data
    train_split, val_split = handler.split_train_val(val_size=0.2)
    
    print("\n" + "="*60)
    print("VALIDATION DATA STATISTICS")
    print("="*60)
    handler.print_statistics(val_split)
    
    # Analyze formats
    logger.info("\nDetecting answer formats...")
    handler.detect_all_formats()
    
    # Example: Get a sample
    print("\n" + "="*60)
    print("EXAMPLE SAMPLE")
    print("="*60)
    sample = handler.get_sample(0)
    for key, value in sample.items():
        print(f"{key}: {value}")