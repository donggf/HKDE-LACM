import os
import argparse
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import torch


def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(examples["sequence"], truncation=True, padding=True, max_length="max_length")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DNA-BERT model.")

    args = parser.parse_args()

    data_files = {
        "train": "data/sample_data/processed/train.tsv",
        "dev": "data/sample_data/processed/dev.tsv",
        "test": "data/sample_data/processed/test.tsv",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("DNABERT_2_117M", trust_remote_code=True)
    # model = AutoModel.from_pretrained("DNABERT_2_117M", trust_remote_code=True, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("DNABERT_2_117M", trust_remote_code=True, num_labels=2)

    # Data encoding
    def encode(examples):
        tokenized = tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=400,
        )
        tokenized["labels"] = examples["label"]  # 关键
        return tokenized

    encoded_dataset = dataset.map(encode, batched=True)
    print(encoded_dataset["train"][0].keys())

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="data/sample_data/finetuned_model_DNABERT_2",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=6,
        # fp16=True,
        save_steps=2000,
        eval_steps=2000,
        warmup_steps=50,
        logging_steps=100,
        overwrite_output_dir=True,
        log_level="info",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],  
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(encoded_dataset["test"])
    print(test_results)

    # Save model
    model.save_pretrained("data/sample_data/finetuned_DNABERT_2")

if __name__ == "__main__":
    main()
