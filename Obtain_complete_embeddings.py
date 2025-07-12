import pandas as pd
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Define the Bi-LSTM model.
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        sequence_embedding = outputs.mean(dim=1)
        return sequence_embedding


# Initialize the Bi-LSTM.
input_dim = 768
hidden_dim = 512
model = BiLSTM(input_dim, hidden_dim)


# Define the dataset and DataLoader.
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# All the files generated in this process are stored in the `target_dictionary` directory.
target_dictionary = "data/sample_data/result/"

for file_name in ['train','dev','test']:

    # Load the data
    file_path = target_dictionary + file_name + "_correct_predictions.csv"
    data = pd.read_csv(file_path).iloc[:,0:769]
    data.iloc[0:,1:769] = data.iloc[0:,1:769].astype(float).round(8)
    # print("data shape: ", data.shape,data)

    # Data preprocessing
    data.columns = ["name"] + [f"feature_{i}" for i in range(1, 769)]
    data["sequence_features"] = data.iloc[:, 1:769].values.tolist()
    
    # Group by `name`, and stack the fragments of each sequence into a matrix
    grouped_data = (data.groupby("name")["sequence_features"].apply(lambda group: torch.tensor(list(group))))

    # Create the dataset and DataLoader.
    sequence_names = grouped_data.index.tolist()
    sequences = list(grouped_data)
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))

    # Compute the sequence context embeddings.
    model.eval()
    sequence_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = model(batch)
            sequence_embeddings.append(embeddings)

    # Concatenate the embeddings of all sequences.
    sequence_embeddings = torch.cat(sequence_embeddings, dim=0)

    # Save the embeddings to a TSV file.
    output_path = target_dictionary + file_name + "_correct_sequence_embeddings.tsv"

    with open(output_path, "w") as f:
        f.write("name\tembedding\n")
        for name, embedding in zip(sequence_names, sequence_embeddings):
            embedding_str = ",".join(map(str, embedding.cpu().numpy()))
            f.write(f"{name}\t{embedding_str}\n")
    print(file_name + "_correct_sequence_embeddings.tsv " + f"have been saved to {output_path}")

# Concatenate the three matrices vertically into a single matrix:
train_data = pd.read_csv("data/sample_data/result/train_correct_sequence_embeddings.tsv",sep='\t')
test_data = pd.read_csv("data/sample_data/result/test_correct_sequence_embeddings.tsv", sep='\t')
dev_data = pd.read_csv("data/sample_data/result/dev_correct_sequence_embeddings.tsv", sep='\t')
merged_data = pd.concat([train_data,dev_data], ignore_index=True)


# Parse the embedding vectors.
def parse_embedding(embedding_str):
    return list(map(float, embedding_str.split(",")))


# Divide the sequence into 1024 features:
# Load the file.
output_path_train =  "data/sample_data/train/result/merged_correct_sequence_embeddings_1024.tsv"
output_path_test = "data/sample_data/test/result/merged_correct_sequence_embeddings_1024.tsv"
data = merged_data
if "embedding" not in data.columns:
    raise ValueError("There is no embedding vector column in the file: 'embedding'")

# Parse the embedding vectors.
def parse_embedding(embedding_str):
    return list(map(float, embedding_str.split(",")))

data["embedding"] = data["embedding"].apply(parse_embedding)
embedding_length = len(data["embedding"].iloc[0])
if not all(len(vec) == embedding_length for vec in data["embedding"]):
    raise ValueError("The length of the embedding vectors is inconsistent.")

# Split the embedding vector column into 1024 feature columns.
embedding_columns = [f"feature_{i}" for i in range(embedding_length)]
embedding_features = pd.DataFrame(data["embedding"].tolist(), columns=embedding_columns)

# Merge the feature columns with the original name column.
result = pd.concat([data["name"], embedding_features], axis=1)
result.to_csv(output_path_train, sep="\t", index=False)
print(f"The conversion is complete, and the features have been saved to {output_path_train}")



#test_data
if "embedding" not in test_data.columns:
    raise ValueError("There is no embedding vector column in the file: 'embedding'")

test_data["embedding"] = test_data["embedding"].apply(parse_embedding)
embedding_length = len(test_data["embedding"].iloc[0])
if not all(len(vec) == embedding_length for vec in test_data["embedding"]):
    raise ValueError("The length of the embedding vectors is inconsistent.")

# Split the embedding vector column into 1024 feature columns.
embedding_columns = [f"feature_{i}" for i in range(embedding_length)]
embedding_features = pd.DataFrame(test_data["embedding"].tolist(), columns=embedding_columns)

# Merge the feature columns with the original name column.
result = pd.concat([test_data["name"], embedding_features], axis=1)
result.to_csv(output_path_test, sep="\t", index=False)
print(f"The conversion is complete, and the features have been saved to {output_path_test}")

