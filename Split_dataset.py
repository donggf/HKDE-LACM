import os
import pandas as pd
import tqdm
import random


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

directory_path = "data/sample_data/data/"

files = []
# Get the files in a directory
entries = os.listdir(directory_path)

for entry in entries:
    if os.path.isfile(os.path.join(directory_path, entry)):
        if entry[0] == 'A' or entry[0] == 'B':
            files.append(entry)


files_sorted = sorted(files, key=lambda x: x[0].lower())
# print(files_sorted)
filename_list = []
label_list = []
for filename in files_sorted:
    filename = filename.split('.')[0]
    if filename == "" or filename == "kmer":
        continue
    if filename[0] == 'A':
        label_list.append("1")
        filename_list.append(filename)
    elif filename[0] == 'B':
        label_list.append("0")
        filename_list.append(filename)
    else:
        continue
print("filename_list",filename_list,type(filename_list))
print("label_list",label_list,type(label_list))

# Split the data and labels into training, validation, and test sets with an 8:1:1 ratio,
# ensuring that the proportion of positive and negative samples is roughly consistent.
def stratified_split_unequal(data, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
    if len(data) != len(labels):
        raise ValueError("The lengths of data and labels must be the same.")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 1.")

    if seed is not None:
        random.seed(seed)

    # Divide the samples into positive and negative based on the labels.
    positive = [(x, y) for x, y in zip(data, labels) if y == '1']
    negative = [(x, y) for x, y in zip(data, labels) if y == '0']
    print("positive:", len(positive), "negative:", len(negative))

    def split_group(group):
        random.shuffle(group)
        total = len(group)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        return group[:train_end], group[train_end:val_end], group[val_end:]

    # Split the positive and negative samples separately.
    pos_train, pos_val, pos_test = split_group(positive)
    neg_train, neg_val, neg_test = split_group(negative)

    # Combine positive and negative samples.
    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    # Shuffle the training set, validation set, and test set.
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Split the data and labels separately.
    train_data, train_labels = zip(*train) if train else ([], [])
    val_data, val_labels = zip(*val) if val else ([], [])
    test_data, test_labels = zip(*test) if test else ([], [])

    return list(train_data), list(val_data), list(test_data), list(train_labels), list(val_labels), list(test_labels)

train_data, val_data, test_data, train_labels, val_labels, test_labels = stratified_split_unequal(filename_list, label_list, seed=42)

print("Training set:", train_data)
print("Training label:", train_labels)
print("Validation set:", val_data)
print("Validation label:", val_labels)
print("Test set:", test_data)
print("Test label:", test_labels)

#Generate k-mers from the sequence (sliding window).
def generate_kmers(sequence, k=6):
    return " ".join(sequence[i:i + k] for i in range(len(sequence) - k + 1))

# Extract 6-mers and write them to the corresponding TSV file
def process_files(input_dir, output_dir,train_data, val_data, test_data, train_labels, val_labels, test_labels,stride = 256,k=6,max_len = 512):
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "train.tsv")
    val_file = os.path.join(output_dir, "val.tsv")
    test_file = os.path.join(output_dir, "test.tsv")

    for file in [train_file, val_file, test_file]:
        with open(file, 'w') as f:
            f.write("name\tsequence\tlabel\n")
            
    print("Split the data into segments and save them:")
    
    for txt_file in train_data:
        txt_name = txt_file
        print("txt_name:",txt_name)
        txt_file = txt_name + ".fa"
        file_path = os.path.join(input_dir, txt_file)

        output_file = train_file
        index = train_data.index(txt_name)
        label = train_labels[index]

        # Read the file and process it.
        sequence = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith(">"):  # Skip lines that start with '>'.
                    sequence.append(line)

            # Combine all lines to generate the complete sequence.
        full_sequence = "".join(sequence)

        for i in range(0, len(full_sequence), stride):
            sub_seq = full_sequence[i:i + max_len]
            kmer_sequence = generate_kmers(sub_seq, k=k)
            with open(output_file, 'a') as f:
                f.write(f"{txt_name}\t{kmer_sequence}\t{label}\n")
            if len(sub_seq) < max_len:
                print("The remaining fragment is insufficient and will not be split further.")
                break
        print(f"Processed file: {txt_file}，write to {output_file}")

    for txt_file in val_data:
        txt_name = txt_file
        print("txt_name:",txt_name)
        txt_file = txt_name + ".fa"
        file_path = os.path.join(input_dir, txt_file)

        output_file = val_file
        index = val_data.index(txt_name)
        label = val_labels[index]

        sequence = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith(">"):
                    sequence.append(line)
        full_sequence = "".join(sequence)

        for i in range(0, len(full_sequence), stride):
            sub_seq = full_sequence[i:i + max_len]
            kmer_sequence = generate_kmers(sub_seq, k=k)
            with open(output_file, 'a') as f:
                f.write(f"{txt_name}\t{kmer_sequence}\t{label}\n")
            if len(sub_seq) < max_len:
                print("The remaining fragment is insufficient and will not be split further.")
                break
        print(f"Processed file: {txt_file}，write to {output_file}")

    for txt_file in test_data:
        txt_name = txt_file
        print("txt_name:",txt_name)
        txt_file = txt_name + ".fa"
        file_path = os.path.join(input_dir, txt_file)

        output_file = test_file
        index = test_data.index(txt_name)
        label = test_labels[index]

        sequence = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith(">"):
                    sequence.append(line)

        full_sequence = "".join(sequence)

        for i in range(0, len(full_sequence), stride):
            sub_seq = full_sequence[i:i + max_len]
            kmer_sequence = generate_kmers(sub_seq, k=k)
            with open(output_file, 'a') as f:
                f.write(f"{txt_name}\t{kmer_sequence}\t{label}\n")
            if len(sub_seq) < max_len:
                print("The remaining fragment is insufficient and will not be split further.")
                break
        print(f"Processed file: {txt_file}，write to {output_file}")



input_directory = "data/sample_data/data"
output_directory = "data/sample_data/result"
process_files(input_directory, output_directory,train_data, val_data, test_data, train_labels, val_labels, test_labels)

train_data = pd.read_csv("data/sample_data/result/train.tsv",sep='\t')
train_data.iloc[:,1:].to_csv("data/sample_data/processed/train.tsv",sep='\t')
print("train_data:",train_data.iloc[:,1:].shape,train_data.iloc[:,1:])
val_data = pd.read_csv("data/sample_data/result/val.tsv",sep='\t')
val_data.iloc[:,1:].to_csv("data/sample_data/processed/val.tsv",sep='\t')
print("val_data:",val_data.iloc[:,1:].shape,val_data.iloc[:,1:])
test_data = pd.read_csv("data/sample_data/result/test.tsv",sep='\t')
test_data.iloc[:,1:].to_csv("data/sample_data/processed/test.tsv",sep='\t')
print("test_data:",test_data.iloc[:,1:].shape,test_data.iloc[:,1:])
print("Conversion completed.")
