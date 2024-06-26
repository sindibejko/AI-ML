# 1. Imports
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


# 2. Functions
def tokenize_function(examples):
    """
    Tokenizes text inputs for NLP model processing.

    Parameters
    ----------
    examples : dict
        Text inputs to be tokenized.

    Returns
    -------
    dict
        Tokenized text inputs.
    """
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


def compute_metrics(pred):
    """
    Computes evaluation metrics for model predictions.

    Parameters
    ----------
    pred : tuple
        Model predictions and true labels.

    Returns
    -------
    dict
        Dictionary of accuracy, precision, recall, and F1 (macro) scores.
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1_macro = f1_score(labels, predictions, average="macro")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1_macro}


# 3. Main Script
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'CUDA: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

# Load dataset
data = load_dataset("rotten_tomatoes")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

# Tokenize the dataset
tokenized_datasets = data.map(tokenize_function, batched=True)

# Initialize model and send it to the appropriate device
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=2).to(device)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results_deberta_large',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=torch.cuda.is_available(),
    warmup_steps=53,
    weight_decay=0.01,
    learning_rate=1e-5,
    logging_dir='./logs_deberta_large',
    logging_steps=250,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    seed=42
)


# Initialize data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Initialize the trainer with the callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate(tokenized_datasets["test"])

results_report = f"""
Evaluation Results on Test Set:
- Loss: {results['eval_loss']:.4f}
- Accuracy: {results['eval_accuracy'] * 100:.2f}%
- F1 Score: {results['eval_f1']:.4f} (Macro)
- Precision: {results['eval_precision']:.4f} (Macro)
- Recall: {results['eval_recall']:.4f} (Macro)
"""

print(results_report)

# Predictions
predictions = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
indices = range(len(pred_labels))

# Create a Dataframe of predictions
df = pd.DataFrame({'index': indices, 'pred': pred_labels})
df.to_csv('results.csv', index=False)