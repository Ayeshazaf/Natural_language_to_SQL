import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_dataset

os.environ["WANDB_DISABLED"] = "true"  # disable wandb

# -----------------------------
# Schema handling
# -----------------------------
def get_schema(): 
    schema = """ Table: customers (id, name, age, city, revenue); Table: students (id, name, age, grade, major); Table: orders (id, customer_id, product, amount, order_date); Table: employees (id, name, department, salary, hire_date); Table: products (id, name, category, price, stock); Table: departments (id, name, location); """ 
    return schema.strip() 

def combine_query_with_schema(nl_query, use_schema=True):
   schema = get_schema() if use_schema else "" 
   return f"Schema: {schema} Query: {nl_query}"

# -----------------------------
# Load data
# -----------------------------
def load_data():
    df = load_dataset("xlangai/spider")["train"].select_columns(["question", "query"])
    df = df.select(range(100))   # use 1000 samples for quicker training
    df = pd.DataFrame(df)
    return df

# -----------------------------
# Train/Val/Test split
# -----------------------------
def split_dataset(df):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

# -----------------------------
# Preprocess (tokenize question -> SQL)
# -----------------------------
def preprocess_data(train_df, val_df, tokenizer, max_input_len=128, max_target_len=128):
    def encode_texts(df):
        inputs = [combine_query_with_schema(q) for q in df["question"].tolist()]
        encodings = tokenizer(
            inputs,
            max_length=max_input_len,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            df["query"].tolist(),
            max_length=max_target_len,
            truncation=True,
            padding="max_length"
        )["input_ids"]

        pad_id = tokenizer.pad_token_id
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in labels]

        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })

    train_dataset = encode_texts(train_df)
    val_dataset = encode_texts(val_df)
    return train_dataset, val_dataset

# -----------------------------
# Train CodeT5 (Seq2Seq)
# -----------------------------
def train_model(train_dataset, val_dataset, model, tokenizer):
    training_args = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",  # validate only at end of each epoch
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   # validation for loss only
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    return trainer

# -----------------------------
# Inference: NL question → SQL
# -----------------------------
def predict_sql(model, tokenizer, question, max_len=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input_text = combine_query_with_schema(question)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_len, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Evaluate exact-match only after training
# -----------------------------
def evaluate_exact_match(model, tokenizer, df):
    correct = 0
    for _, row in df.iterrows():
        question = row["question"]
        gold_sql = row["query"].strip()
        pred_sql = predict_sql(model, tokenizer, question).strip()
        if pred_sql == gold_sql:
            correct += 1
    total = len(df)
    exact_match_score = correct / total if total > 0 else 0
    print(f"Exact-match score: {exact_match_score:.2f} ({correct}/{total})")
    return exact_match_score

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    train_df, val_df, test_df = split_dataset(df)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

    train_dataset, val_dataset = preprocess_data(train_df, val_df, tokenizer)

    trainer = train_model(train_dataset, val_dataset, model, tokenizer)

    # Quick test
    example = "Show me the names of all students older than 20."
    sql = predict_sql(model, tokenizer, example)
    print("NL Query:", example)
    print("Generated SQL:", sql)

    # Evaluate exact-match on test set
    evaluate_exact_match(model, tokenizer, test_df)
