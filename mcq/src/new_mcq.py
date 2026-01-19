import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SPLIT = "test"  # "train" or "test"
MAX_NEW_TOKENS = 1  # MCQ = one token decision

# =========================
# HELPERS
# =========================
def extract_choice(text):
    """
    Extract first valid MCQ choice (A/B/C/D).
    Fallback to 'A' if nothing found.
    """
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else "A"


# =========================
# MAIN
# =========================
def main():

    if SPLIT == "train":
        data_path = "../train_dataset_mcq.csv"
        out_path = "../mcq_predictions_train.csv"
    else:
        data_path = "../test_dataset_mcq.csv"
        out_path = "../mcq_predictions_test.csv"

    df = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    predictions = []

    for _, row in df.iterrows():

        prompt = (
            row["prompt"]
            + "\n\n"
            + "Answer with only ONE letter: A, B, C, or D.\n"
            + "Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # only take model continuation
        generated = decoded[len(prompt):].strip()

        choice = extract_choice(generated)
        predictions.append(choice)

    df["prediction"] = predictions
    df.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")
    print(f"Processed {len(df)} questions")


if __name__ == "__main__":
    main()