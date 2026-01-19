import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SPLIT = "test"   # change to "test" later

def main():
    if SPLIT == "train":
        data_path = "../train_dataset_mcq.csv"
        out_path  = "../mcq_predictions_train.csv"
    else:
        data_path = "../test_dataset_mcq.csv"
        out_path  = "../mcq_predictions_test.csv"

    df = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    choices = ["A", "B", "C", "D"]
    choice_ids = tokenizer.convert_tokens_to_ids(choices)

    predictions = []

    for _, row in df.iterrows():
        prompt = (
            row["prompt"].strip()
            + "\nAnswer with only one letter: A, B, C, or D.\nAnswer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, choice_ids]

        pred_idx = torch.argmax(logits).item()
        predictions.append(choices[pred_idx])

    df["prediction"] = predictions
    df.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")
    print(f"Processed {len(df)} questions")

if __name__ == "__main__":
    main()