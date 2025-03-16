from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import tqdm

# Load the fine-tuned model and tokenizer.
tokenizer = BertTokenizer.from_pretrained("DNABERT-6")
model = BertForSequenceClassification.from_pretrained("data/sample_data/finetuned_DNABERT6").to('cuda')

# Load the dataset
data_files = {
    "train": "data/sample_data/processed/train.tsv",
    "validation": "data/sample_data/processed/val.tsv",
    "test": "data/sample_data/processed/test.tsv",
}
dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# Data encoding and batch processing.
def encode(examples):
    return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=512)

encoded_dataset = dataset.map(encode, batched=True)

# Save the results to a file.
def save_results(predictions, embeddings, file_name):
    embeddings = np.array(embeddings)
    embedding_flattened = embeddings.reshape(embeddings.shape[0], -1)
    df = pd.DataFrame(embedding_flattened)
    df["prediction"] = predictions
    df.to_csv(file_name, index=False)

# Obtain predictions and embeddings.
def get_predictions_and_embeddings(dataset, model, tokenizer, batch_size=10000, file_name_prefix="data/sample_data/results"):
    predictions = []
    embeddings = []
    sequences = []
    count = 0

    # Set the model to evaluation mode.
    model.eval()

    for idx, example in tqdm.tqdm(enumerate(dataset),total = len(dataset)):
        inputs = tokenizer(example["sequence"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Obtain the predicted labels.
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Obtain the embeddings.
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1].squeeze(0)  # Extract the embeddings from the last layer.
        pooled_embedding = torch.mean(last_hidden_state, dim=0).unsqueeze(0).cpu().numpy()
        embeddings.append(pooled_embedding)
        # embeddings.append(last_hidden_state.cpu().numpy())

        # Save the sequence context and predicted labels.
        sequences.append(example["sequence"])
        predictions.append(predicted_label)

        # After processing every 'batch_size' samples (e.g., 10,000), write to the file and release memory.
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(dataset):
            save_results(predictions, embeddings, f"{file_name_prefix}_{count}.csv")
            count += 1

            # Reset the list to release memory.
            sequences = []
            predictions = []
            embeddings = []

            # Clean up GPU memory.
            torch.cuda.empty_cache()
    return sequences, predictions, embeddings

# Obtain the predictions and embeddings for the training set, validation set, and test set.
train_sequences, train_predictions, train_embeddings = get_predictions_and_embeddings(encoded_dataset["train"], model, tokenizer, batch_size=15000, file_name_prefix="data/sample_data/embedding_vector/train_predictions_and_embeddings")
val_sequences, val_predictions, val_embeddings = get_predictions_and_embeddings(encoded_dataset["validation"], model, tokenizer, batch_size=15000, file_name_prefix="data/sample_data/embedding_vector/val_predictions_and_embeddings")
test_sequences, test_predictions, test_embeddings = get_predictions_and_embeddings(encoded_dataset["test"], model, tokenizer, batch_size=15000, file_name_prefix="data/sample_data/embedding_vector/test_predictions_and_embeddings")


print("Predictions and embeddings saved to CSV files.")

