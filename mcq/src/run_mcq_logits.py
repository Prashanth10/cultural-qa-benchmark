import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SPLIT = "test"   # change to "test" for final submission

TRAIN_PATH = "../train_dataset_mcq.csv"
TEST_PATH  = "../test_dataset_mcq.csv"

OUT_TRAIN = "../mcq_predictions_train.csv"
OUT_TEST  = "../mcq_predictions_test.csv"


def score_completion(model, input_ids, completion_ids):
    """
    Teacher-forcing log-prob of completion_ids given input_ids.
    Returns total log-prob (float).
    """
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids + completion_ids], device=device)

    with torch.no_grad():
        out = model(ids)
        logits = out.logits  # [1, seq, vocab]
        logprobs = torch.log_softmax(logits, dim=-1)

    # log P(token_t | tokens_<t>) for completion tokens
    # completion starts at position len(input_ids)
    start = len(input_ids)
    total = 0.0
    for i, tok_id in enumerate(completion_ids):
        pos = start + i - 1  # token predicted at previous position
        total += logprobs[0, pos, tok_id].item()
    return total


def pick_choice(model, tokenizer, prompt):
    # Make the prompt end right before the answer
    clean = prompt.strip()
    if "Answer:" in clean:
        clean = clean.split("Answer:")[0].strip()

    clean += "\nAnswer with only one letter: A, B, C, or D.\nAnswer:"

    input_ids = tokenizer.encode(clean, add_special_tokens=False)

    # Score " A", " B", ... (leading space helps many tokenizers)
    options = {}
    for ch in ["A", "B", "C", "D"]:
        comp = " " + ch
        comp_ids = tokenizer.encode(comp, add_special_tokens=False)
        options[ch] = score_completion(model, input_ids, comp_ids)

    return max(options, key=options.get)


def main():
    data_path = TRAIN_PATH if SPLIT == "train" else TEST_PATH
    out_path  = OUT_TRAIN if SPLIT == "train" else OUT_TEST

    df = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    preds = []
    for _, row in df.iterrows():
        preds.append(pick_choice(model, tokenizer, row["prompt"]))

    df["prediction"] = preds
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | n={len(df)}")


if __name__ == "__main__":
    main()