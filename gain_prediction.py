from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,BertForSequenceClassification,BertTokenizer,AutoConfig
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import tqdm
import torch.nn.functional as F
from pathlib import Path
import shutil

# Source and target directories
src_dir = Path("DNABERT_2_117M")
dst_dir = Path("data/sample_data/finetuned_DNABERT_2")

# Ensure target directory exists
dst_dir.mkdir(parents=True, exist_ok=True)

# Files to skip
skip_files = {"pytorch_model.bin", "config.json"}

# Copy all files except those in skip_files
for item in src_dir.iterdir():
    if item.is_file() and item.name not in skip_files:
        shutil.copy(item, dst_dir / item.name)
        print(f"Copied: {item.name}")


tokenizer = AutoTokenizer.from_pretrained("DNABERT_2_117M")
config = AutoConfig.from_pretrained("data/sample_data/finetuned_DNABERT_2",trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("data/sample_data/finetuned_DNABERT_2",config=config,trust_remote_code=True).to('cuda')


data_files = {
    "train": "data/sample_data/processed/train.tsv",
    "validation": "data/sample_data/processed/dev.tsv",
    "test": "data/sample_data/processed/test.tsv",
}
dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

def encode(examples):
    return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=200)

encoded_dataset = dataset.map(encode, batched=True)

# Save the results to a file
def save_results(probabilities_1, probabilities_0, embeddings, file_name):
    embeddings = np.array(embeddings)
    #
    # The embedding of each example is a vector, usually of shape (512, hidden_size), which you can flatten into one dimension
    embedding_flattened = embeddings.reshape(embeddings.shape[0], -1)

    df = pd.DataFrame(embedding_flattened)
    # df["sequence"] = sequences
    df["probabilities_1"] = probabilities_1
    df["probabilities_0"] = probabilities_0
    df.to_csv(file_name, index=False)

# Functions to get predictions and embeddings
def get_probabilities_and_embeddings(dataset, model, tokenizer, batch_size=10000, file_name_prefix="data/sample_data/results"):
    probabilities_0 = []
    probabilities_1 = []
    embeddings = []
    sequences = []
    count = 0

    model.eval()

    for idx, example in tqdm.tqdm(enumerate(dataset),total = len(dataset)):
        inputs = tokenizer(example["sequence"], return_tensors="pt", truncation=True, padding="max_length", max_length=200)

        inputs = {key: value.to('cuda') for key, value in inputs.items()}  # 移动到 GPU

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        logits = outputs.logits
        probability = F.softmax(logits, dim=-1)
        prob_1 = probability[0].tolist()[1]
        prob_0 = probability[0].tolist()[0]

        # Get embeddings
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1].squeeze(0)
        pooled_embedding = torch.mean(last_hidden_state, dim=0).unsqueeze(0).cpu().numpy()  # 形状变为 (1, hidden_size)
        embeddings.append(pooled_embedding)

        # Save sequence context and prediction labels
        sequences.append(example["sequence"])
        probabilities_1.append(prob_1)
        probabilities_0.append(prob_0)

        # After processing batch_size (e.g. 15000) samples, write the file and release the memory
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(dataset):
            save_results(probabilities_1, probabilities_0, embeddings, f"{file_name_prefix}_{count}.csv")
            count += 1

            # Reset the list to free up memory
            sequences = []
            probabilities_1 = []
            probabilities_0 = []
            embeddings = []

            # Cleaning up GPU memory
            torch.cuda.empty_cache()
    	
    return sequences, probabilities_1, probabilities_0, embeddings

# Get predictions and embeddings for the training, validation, and test sets
train_sequences, train_probabilities_1, train_probabilities_0, train_embeddings = get_probabilities_and_embeddings(encoded_dataset["train"], model, tokenizer, batch_size=15000, file_name_prefix="data/sample_data/embedding_vector/train_probabilities_and_embeddings")
dev_sequences, dev_probabilities_1, dev_probabilities_0, dev_embeddings = get_probabilities_and_embeddings(encoded_dataset["validation"], model, tokenizer, batch_size=15000, file_name_prefix="data/sample_data/embedding_vector/dev_probabilities_and_embeddings")
test_sequences, test_probabilities_1, test_probabilities_0, test_embeddings = get_probabilities_and_embeddings(encoded_dataset["test"], model, tokenizer, batch_size=15000, file_name_prefix="data/sample_data/embedding_vector/test_probabilities_and_embeddings")


print("probabilities and embeddings saved to CSV files.")
