# Fine-tuning AraBERT on AraNER Dataset for Named Entity Recognition

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from torch.optim import AdamW  # Import AdamW from torch.optim instead
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and parameters
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SAVE_PATH = "./arabert_ner_model"

# Load the AraNER dataset
def load_araner_dataset():
    try:
        # Attempt to load directly from Hugging Face datasets
        dataset = load_dataset("asas-ai/ANERCorp")
        print("Successfully loaded AraNER dataset from Hugging Face")
        return dataset
    except:
        print("Could not load AraNER from Hugging Face, trying local path...")

        # If not available through HF, load from local files
        # You would need to have the dataset files downloaded
        # Adjust paths as needed
        train_df = pd.read_csv("araner/train.txt", sep="\t", header=None, names=["word", "tag"])
        val_df = pd.read_csv("araner/dev.txt", sep="\t", header=None, names=["word", "tag"])
        test_df = pd.read_csv("araner/test.txt", sep="\t", header=None, names=["word", "tag"])

        # Convert to datasets format
        # This is a simplified implementation - you may need to adjust for the actual format
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }

# Load and prepare the dataset
dataset = load_araner_dataset()

# Print available splits
print("Available splits in the dataset:", dataset.keys())

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get the label list from the dataset
# Assuming dataset contains all possible labels in the training set
label_list = list(set(dataset["train"]["tag"])) if isinstance(dataset["train"], pd.DataFrame) else list(set(dataset["train"]["tag"]))
label_list.sort()  # Sort for determinism
label_encoding = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

print(f"Number of labels: {num_labels}")
print(f"Labels: {label_list}")

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id that is None
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and prepare the dataset
if "datasets" in str(type(dataset)):
    # HuggingFace datasets
    def prepare_dataset(examples):
        # Ensure that tokens and tags are lists of lists
        tokens = [[word] for word in examples["word"]]
        tags = [[label_encoding[tag]] for tag in examples["tag"]]

        return {"tokens": tokens, "ner_tags": tags}

    dataset = dataset.map(prepare_dataset, batched=True)

    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
else:
    # Custom implementation for DataFrame format
    # This would need to be adjusted based on actual format
    # This is just a placeholder for the approach
    def prepare_dataset(df):
        tokens = df.groupby("sentence_id")["word"].apply(list).tolist()
        tags = df.groupby("sentence_id")["tag"].apply(lambda x: [label_encoding[l] for l in x]).tolist()

        return tokenize_and_align_labels({
            "tokens": tokens,
            "ner_tags": tags
        })

    tokenized_datasets = {
        "train": prepare_dataset(dataset["train"]),
        "validation": prepare_dataset(dataset["validation"]),
        "test": prepare_dataset(dataset["test"])
    }

# Load the model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    eval_strategy="epoch",  # Corrected argument name
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

# Define Trainer
eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# Evaluate on test set
print("Evaluating on test set...")
results = trainer.evaluate(tokenized_datasets["test"])
print(results)

# Example usage of the fine-tuned model
def predict_ner(text):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert token predictions to word predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_predictions = [label_list[prediction] for prediction, token in zip(predictions[0].cpu().numpy(), tokens)
                        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]]

    # Since we don't have word_ids, we'll just return token predictions
    return token_predictions

# Test with sample text
sample_text = "قال الرئيس المصري عبد الفتاح السيسي في القاهرة اليوم"
print(f"Sample text: {sample_text}")
predictions = predict_ner(sample_text)
print(f"Predictions: {predictions}")

if __name__ == "__main__":
    print("Fine-tuning complete!")
