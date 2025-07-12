import os
import pandas as pd
import tqdm
import random
from pathlib import Path
import shutil

def stratified_split_unequal(data, labels, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, seed=None):

    if len(data) != len(labels):
        raise ValueError("Data and labels must be the same length!")
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train_ratio, dev_ratio, and test_ratio must add up to 1!")

    if seed is not None:
        random.seed(seed)

    # Divide into positive samples and negative samples according to labels
    positive = [(x, y) for x, y in zip(data, labels) if y == 1]
    negative = [(x, y) for x, y in zip(data, labels) if y == 0]
    print("positive:", len(positive), "negative:", len(negative))

    def split_group(group):
        random.shuffle(group)
        total = len(group)
        train_end = int(total * train_ratio)
        dev_end = train_end + int(total * dev_ratio)
        return group[:train_end], group[train_end:dev_end], group[dev_end:]

    pos_train, pos_dev, pos_test = split_group(positive)
    neg_train, neg_dev, neg_test = split_group(negative)

    train = pos_train + neg_train
    dev = pos_dev + neg_dev
    test = pos_test + neg_test

    train_data, train_labels = zip(*train) if train else ([], [])
    dev_data, dev_labels = zip(*dev) if dev else ([], [])
    test_data, test_labels = zip(*test) if test else ([], [])
    print("train_data:",train_data,train_labels,"dev_data:",dev_data,dev_labels,"test_data:",test_data,test_labels)

    return list(train_data), list(dev_data), list(test_data), list(train_labels), list(dev_labels), list(test_labels)




def process_files(input_dir, output_dir,train_data, dev_data, test_data, train_labels, dev_labels, test_labels,stride = 200,max_len = 800):

    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "train.tsv")
    dev_file = os.path.join(output_dir, "dev.tsv")
    test_file = os.path.join(output_dir, "test.tsv")

    for file in tqdm.tqdm([train_file, dev_file, test_file]):
        with open(file, 'w') as f:
            f.write("name\tsequence\tlabel\n")  # Write column names


    for txt_file in tqdm.tqdm(train_data):
        txt_name = txt_file
        print("txt_name:",txt_name)
        txt_file = txt_name + ".fa"
        file_path = os.path.join(input_dir, txt_file)

        output_file = train_file
        index = train_data.index(txt_name)
        label = train_labels[index]

        sequence = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith(">"):
                    sequence.append(line)

        full_sequence = "".join(sequence)

        for i in range(0, len(full_sequence), stride):
            sub_seq = full_sequence[i:i + max_len]

            with open(output_file, 'a') as f:
                f.write(f"{txt_name}\t{sub_seq}\t{label}\n")
            if len(sub_seq) < max_len:
                print("Insufficient fragments left, no more segmentation!")
                print(f"Processed file: {txt_file}, written to {output_file}.")
                break

    for txt_file in tqdm.tqdm(dev_data):
        txt_name = txt_file
        print("txt_name:", txt_name)
        txt_file = txt_name + ".fa"
        file_path = os.path.join(input_dir, txt_file)

        output_file = dev_file
        index = dev_data.index(txt_name)
        label = dev_labels[index]

        sequence = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith(">"):
                    sequence.append(line)

        full_sequence = "".join(sequence)

        for i in range(0, len(full_sequence), stride):
            sub_seq = full_sequence[i:i + max_len]

            with open(output_file, 'a') as f:
                f.write(f"{txt_name}\t{sub_seq}\t{label}\n")
            if len(sub_seq) < max_len:
                print("Insufficient fragments left, no more segmentation!")
                print(f"Processed file: {txt_file}, written to {output_file}.")
                break

    for txt_file in tqdm.tqdm(test_data):
        txt_name = txt_file
        print("txt_name:", txt_name)
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

            with open(output_file, 'a') as f:
                f.write(f"{txt_name}\t{sub_seq}\t{label}\n")
            if len(sub_seq) < max_len:
                print("Insufficient fragments left, no more segmentation!")
                print(f"Processed file: {txt_file}, written to {output_file}.")
                break



current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

directory_path = "data/sample_data/data"

files = []
for root, dirs, files_in_dir in os.walk(directory_path):
    for file in files_in_dir:
        if file[0] == 'A' or file[0] == 'B':
            files.append(os.path.join(file))

files_sorted = sorted(files, key=lambda x: x[0].lower())
print(files_sorted)
filename_list = []
label_list = []

for filename in files_sorted:
    filename = filename.split('.')[0]
    if filename == "" or filename == "kmer":
        continue
    if filename[0] == 'A':
        label_list.append(1)
        filename_list.append(filename)
    elif filename[0] == 'B':
        label_list.append(0)
        filename_list.append(filename)
    else:
        continue

print("filename_list",filename_list,type(filename_list))
print("label_list",label_list,type(label_list))

train_data, dev_data, test_data, train_labels, dev_labels, test_labels = stratified_split_unequal(filename_list, label_list, seed=42)

print("training set:", train_data)
print("training label:", train_labels)
print("Validation set:", dev_data)
print("Validation label:", dev_labels)
print("test set:", test_data)
print("test label:", test_labels)


base_src_dir = Path("data/sample_data/data")
train_dest = Path("data/sample_data/train")
test_dest = Path("data/sample_data/test")

train_dest.mkdir(parents=True, exist_ok=True)
test_dest.mkdir(parents=True, exist_ok=True)

# Merge filenames for training and validation
to_train = set(train_data) | set(dev_data)
to_test = set(test_data)

# Traverse source directory and copy matching files
copied_train, copied_test = 0, 0

for src_file in base_src_dir.iterdir():
    if not src_file.is_file():
        continue
    stem = src_file.stem  
    if stem in to_train:
        shutil.copy(src_file, train_dest / src_file.name)
        copied_train += 1
    elif stem in to_test:
        shutil.copy(src_file, test_dest / src_file.name)
        copied_test += 1

print(f"Copied {copied_train} files to {train_dest}")
print(f"Copied {copied_test} files to {test_dest}")

input_directory = "data/sample_data/data"  # Enter directory path
output_directory = "data/sample_data/result"  # Output directory path
process_files(input_directory, output_directory,train_data, dev_data, test_data, train_labels, dev_labels, test_labels)

train_data = pd.read_csv("data/sample_data/result/train.tsv",sep='\t')
train_shuffle = train_data.sample(frac=1).reset_index(drop=True)
train_shuffle.to_csv("data/sample_data/result/train_shuffle.tsv",sep='\t')
train_shuffle.iloc[:,1:].to_csv("data/sample_data/processed/train.tsv",sep='\t')
print("train_data:",train_shuffle.iloc[:,1:].shape,train_shuffle.iloc[:,1:])
dev_data = pd.read_csv("data/sample_data/result/dev.tsv",sep='\t')
dev_shuffle = dev_data.sample(frac=1).reset_index(drop=True)
dev_shuffle.to_csv("data/sample_data/result/dev_shuffle.tsv",sep='\t')
dev_shuffle.iloc[:,1:].to_csv("data/sample_data/processed/dev.tsv",sep='\t')
print("dev_data:",dev_shuffle.iloc[:,1:].shape,dev_shuffle.iloc[:,1:])
test_data = pd.read_csv("data/sample_data/result/test.tsv",sep='\t')
test_shuffle = test_data.sample(frac=1).reset_index(drop=True)
test_shuffle.to_csv("data/sample_data/result/test_shuffle.tsv",sep='\t')
test_shuffle.iloc[:,1:].to_csv("data/sample_data/processed/test.tsv",sep='\t')
print("test_data:",test_shuffle.iloc[:,1:].shape,test_shuffle.iloc[:,1:])
print("Conversion completed!")
