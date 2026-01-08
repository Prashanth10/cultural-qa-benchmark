import os
import torch
import json
import logging
from typing import List, Dict, Optional, Literal, Union
from pathlib import Path
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from tqdm import tqdm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

logger.info(f"✓ HF_TOKEN loaded: {HF_TOKEN[:8]}...")


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 50
    temperature: float = 0.1  # ✅ LOWERED: More conservative for factual answers
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


class LlamaZeroShotGenerator:
    """Zero-shot answer generator using Meta-Llama-3-8B-Instruct"""
    
    # ✅ UPDATED PROMPTS WITH BEST PRACTICES
    PROMPT_TEMPLATES = {
        "BASIC": """You are a helpful assistant answering cultural questions. 
Answer concisely (1-3 words).

Question: {question}
Answer:""",
        
        "CONTEXT_AWARE": """You are a cultural knowledge expert for {country}.

TASK: Answer with ONLY the answer, no explanation.

FORMAT RULES:
1. Numbers (Arabic numerals): digits only (e.g., "3", "20")
2. Time (HHMM): 4 digits only (e.g., "1800", "0700")
3. Date (MMDD): 4 digits only (e.g., "1225")
4. Names: 1-3 words (e.g., "jack ma")
5. Concepts: 2-5 words (e.g., "lunar new year")
6. If unsure: "UNKNOWN"

Question: {question}
Answer:""",
        
        "FORMAT_AWARE": """You are answering a cultural question.
Required format: {format}

RULES:
- Return ONLY the answer
- NO explanations
- NO "I think" or "probably"
- Match the required format

Question: {question}
Answer:""",
        
        "DETAILED": """You are a cultural knowledge expert for {country}.

TASK: Answer cultural questions with ONLY the answer, no explanation.

ANSWER RULES:
1. Provide ONLY the answer
2. Follow the required format
3. If multiple valid answers, choose the MOST COMMON
4. NO explanations, sources, or extra text
5. If unsure: "UNKNOWN"

Question: {question}
Answer:""",
        
        "FEW_SHOT": """You are a cultural knowledge expert for {country}.

TASK: Answer concisely with ONLY the answer, no explanation.

ANSWER FORMAT RULES:
- Numbers: digits only
- Time (HHMM): 4 digits
- Names: 1-3 words
- Concepts: 2-5 words

EXAMPLES:
Q: What is the most famous sport in {country}?
A: [sport name]

Q: Who is the most famous entrepreneur in {country}?
A: [name]

Q: What is the biggest festival?
A: [festival name]

Q: At what time do people finish work? HHMM format.
A: [4 digits]

CRITICAL: Return ONLY the answer.

Question: {question}
Answer:""",
        
        "PRODUCTION_BEST": """You are a cultural knowledge expert for United States, United Kingdom, China, and Iran.

TASK: Answer cultural questions with ONLY the answer, no explanation.

ANSWER FORMAT RULES:
1. Arabic numerals: digits ONLY (e.g., "3", "20", "1800")
2. Time (HHMM format): 4 digits ONLY (e.g., "1800")
3. Date (MMDD format): 4 digits ONLY (e.g., "1231")
4. Decimal numbers: with decimal point (e.g., "2", "3.5")
5. Names/people: 1-3 words (e.g., "jack ma", "muhammad ali")
6. Concepts/activities: 2-5 words (e.g., "lunar new year", "american football")
7. Single terms: just the word (e.g., "wrestling", "tea", "dogs")

CRITICAL CONSTRAINTS:
- Return ONLY the answer - no explanations
- NO "I think", "probably", or "based on"
- If multiple valid answers, choose the MOST COMMONLY ACCEPTED one
- If completely unsure: respond "UNKNOWN"

DIVERSE EXAMPLES:
Q: What is the most famous traditional sport in the US?
A: american football

Q: How many people are in a typical family in China? Provide Arabic numerals only.
A: 3

Q: At what time do most people finish work in China? Provide in HHMM format.
A: 1800

Q: What is the biggest festival in China?
A: lunar new year

Q: Who is the most famous entrepreneur in China?
A: jack ma

Question: {question}
Answer:"""
    }
    
    COUNTRY_NAMES = {
        "US": "United States",
        "GB": "United Kingdom",
        "CN": "China",
        "IR": "Iran"
    }
    
    FORMAT_NAMES = {
        "TEXT": "plain text (1-3 words)",
        "HHMM": "time format (e.g., 1800, 0900)",
        "MMDD": "date format (e.g., 1225, 0101)",
        "NUMERIC": "numeric (e.g., 5, 25)"
    }
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        use_8bit: bool = True,
        use_flash_attention: bool = True,
        device: str = "auto",
        generation_config: Optional[GenerationConfig] = None
    ):
        """Initialize Llama generator"""
        
        self.model_name = model_name
        self.generation_config = generation_config or GenerationConfig()
        self.device = device
        self.use_8bit = use_8bit
        self.use_flash_attention = use_flash_attention
        
        logger.info(f"Loading model: {model_name}")
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=HF_TOKEN
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        quantization_config = None
        if use_8bit:
            logger.info("Enabling 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        
        logger.info("Loading model weights...")
        try:
            attn_impl = "flash_attention_2" if use_flash_attention else "eager"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=HF_TOKEN,
                quantization_config=quantization_config,
                device_map="auto" if device == "auto" else None,
                attn_implementation=attn_impl,
                torch_dtype=torch.float16 if not use_8bit else None
            )
        except Exception as e:
            logger.warning(f"Could not use flash attention: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=HF_TOKEN,
                quantization_config=quantization_config,
                device_map="auto" if device == "auto" else None,
            )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.model.eval()
        logger.info(f"✓ Model loaded successfully on {self.model.device}")
        
    def generate_single(
        self,
        question: str,
        country: Optional[str] = None,
        answer_format: Optional[str] = None,
        template: Literal["BASIC", "CONTEXT_AWARE", "FORMAT_AWARE", "DETAILED", 
                         "FEW_SHOT", "PRODUCTION_BEST"] = "PRODUCTION_BEST"  # ✅ CHANGED DEFAULT
    ) -> str:
        """Generate single answer for a question"""
        
        if template not in self.PROMPT_TEMPLATES:
            template = "PRODUCTION_BEST"  # ✅ FALLBACK TO BEST
        
        prompt_template = self.PROMPT_TEMPLATES[template]
        country_name = self.COUNTRY_NAMES.get(country, "World")
        format_desc = self.FORMAT_NAMES.get(answer_format, "plain text")
        
        prompt = prompt_template.format(
            question=question,
            country=country_name,
            format=format_desc
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.max_new_tokens,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
                do_sample=self.generation_config.do_sample,
                repetition_penalty=self.generation_config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        answer = self._clean_answer(answer)
        
        return answer
    
    def generate_batch(
        self,
        questions: Union[List[Dict], List[str]],
        template: str = "PRODUCTION_BEST",  # ✅ CHANGED DEFAULT
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate answers for multiple questions
        ✅ FIXED: Now handles both dict and string inputs
        """
        
        answers = []
        logger.info(f"Generating {len(questions)} answers...")
        
        for i in tqdm(range(0, len(questions), batch_size)):
            batch = questions[i:i+batch_size]
            
            for item in batch:
                try:
                    # ✅ FIXED: Handle both dict and string inputs
                    if isinstance(item, dict):
                        question = item.get('question', '')
                        country = item.get('country')
                        answer_format = item.get('format')
                    elif isinstance(item, str):
                        # If it's just a string, use it as question
                        question = item
                        country = None
                        answer_format = None
                    else:
                        logger.error(f"Unexpected item type: {type(item)}")
                        answers.append("")
                        continue
                    
                    answer = self.generate_single(
                        question=question,
                        country=country,
                        answer_format=answer_format,
                        template=template
                    )
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    answers.append("")
        
        logger.info(f"✓ Generated {len(answers)} answers")
        return answers
    
    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Clean generated answer"""
        answer = answer.strip()
        
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
        
        if ':' in answer and len(answer.split(':')[0]) < 20:
            answer = answer.split(':', 1)[1].strip()
        
        words = answer.split()
        if len(words) > 10:
            answer = ' '.join(words[:10])
        
        return answer
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "8bit_quantization": self.use_8bit,
            "total_parameters": f"{total_params:,}",
            "trainable_parameters": f"{trainable_params:,}",
            "generation_config": {
                "temperature": self.generation_config.temperature,
                "max_new_tokens": self.generation_config.max_new_tokens,
                "top_p": self.generation_config.top_p
            }
        }


# Backward compatibility alias
ZeroShotGenerator = LlamaZeroShotGenerator


if __name__ == "__main__":
    print("Testing Llama Generator with BEST PROMPTS...")
    gen = LlamaZeroShotGenerator()
    info = gen.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Device: {info['device']}")
    print(f"Temperature: {info['generation_config']['temperature']}")
    
    answer = gen.generate_single(
        "What is the most famous entrepreneur in China?",
        country="CN",
        template="PRODUCTION_BEST"
    )
    print(f"\nAnswer: {answer}")
    print("✅ Works with BEST PROMPTS!")