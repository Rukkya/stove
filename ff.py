import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# Load the ANERCorp dataset
dataset = load_dataset("asas-ai/ANERCorp")

# Extract unique labels
unique_tags = sorted(set(dataset["train"]["tag"]))
label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2label = {idx: tag for tag, idx in label2id.items()}

# Group words and tags into sentences
def group_sentences(dataset_split):
    sentences = []
    tags = []
    current_sentence = []
    current_tags = []
    for word, tag in zip(dataset_split["word"], dataset_split["tag"]):
        if word == ".":
            current_sentence.append(word)
            current_tags.append(tag)
            sentences.append(current_sentence)
            tags.append(current_tags)
            current_sentence = []
            current_tags = []
        else:
            current_sentence.append(word)
            current_tags.append(tag)
    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)
    return {"tokens": sentences, "ner_tags": tags}

# Apply grouping to train and test splits
train_data = group_sentences(dataset["train"])
test_data = group_sentences(dataset["test"])

# Convert tags to IDs
train_data["ner_tags"] = [[label2id[tag] for tag in seq] for seq in train_data["ner_tags"]]
test_data["ner_tags"] = [[label2id[tag] for tag in seq] for seq in test_data["ner_tags"]]

# Load tokenizer and model
model_checkpoint = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=False)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])  # or -100 to ignore subword tokens
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Prepare datasets
from datasets import Dataset
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

# Define metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[pred] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lab] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Set training arguments
training_args = TrainingArguments(
    output_dir="./arabert-anercorp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("arabert-finetuned-anercorp")
