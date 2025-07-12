import os
import time
import tqdm
import pandas as pd
import joblib
from functools import reduce
from train import FeatureSelector,KmerSelector
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, matthews_corrcoef,
    roc_curve
)

#Obtain the sample names and their corresponding labels.
def get_samples_and_labels(directory_name):
    directory = directory_name + "/test"
    # Get all the file names in the specified path.
    all_files_and_folders = os.listdir(directory)
    file_names = []
    for item in all_files_and_folders:
        if os.path.isfile(os.path.join(directory, item)):
            file_names.append(item)

    label_list = []
    filename_list = []
    for filename in file_names:
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
            break
    print("filename_list:",filename_list,len(filename_list))
    print("label_list:",label_list,len(label_list))
    return filename_list,label_list

#Calculate the k-mers for each sample and aggregate them into a matrix.
def get_kmers(k,directory_name,target_name):
    directory = directory_name + "/test"
    # Calculate the k-mers
    mkdir_command = f"mkdir data/sample_data/test/kmer/{k}mer"
    os.system(mkdir_command)
    print(str(k) +"mer Directory created successfully.")
    print("Calculating k-mers:")
    for name in tqdm.tqdm(filename_list):
        file_name = name
        print("file_name:",file_name)
        #Calculate k-mers using Jellyfish.
        count_command = f"jellyfish count -m {k} -o {directory}/kmer/{k}mer/{file_name}.jf -c 3 -s 1G -t 16 {directory}/{file_name}.fa"
        os.system(count_command)
        time.sleep(4)

        # Export the results using `jellyfish dump` as a TSV file.
        dump_command = f"jellyfish dump -c -t {directory}/kmer/{k}mer/{file_name}.jf > {directory}/kmer/{k}mer/{file_name}.tsv"
        os.system(dump_command)
        time.sleep(2)

    all_reads = reduce(lambda x, y: [t + j for t in x for j in y], [['A', 'C', 'G', 'T']] * k)
    # Initialize an empty DataFrame to store the integrated results.
    merged_df = pd.DataFrame({
        'reads': all_reads,
        'frequency': [0] * len(all_reads)
    })
    i = 0
    folder_path = directory + '/kmer/' + str(k) + 'mer'
    #aggregate them into a matrix.
    for filename in tqdm.tqdm(filename_list):
        file_name = filename + ".tsv"
        filepath = os.path.join(folder_path, file_name)
        if os.path.isfile(filepath):
            sample = pd.read_csv(filepath, sep='\t', header=None,
                                 names=["reads", "frequency"])
            merged_df = pd.merge(merged_df, sample, on='reads', how='outer', suffixes=('_old','_new'))
            merged_df.rename(columns={'frequency_old': 'frequency','frequency_new': filename}, inplace=True)
            i = i + 1
    merged_df.drop('frequency', axis=1, inplace=True)
    merged_df.to_csv(target_name, sep="\t", index=False)
    print("The k-mers feature matrix has been saved at:",target_name)

#Read the embedded sequence.
def get_embedding(X_embedding_path,X_embedding_incorrect_path):
    X_embedding = pd.read_csv(X_embedding_path,sep="\t")
    X_embedding_incorrect = pd.read_csv(X_embedding_incorrect_path,sep="\t")
    print("Shape of the forward embedding vector:",X_embedding.shape)
    print("Shape of the reverse embedding vector:",X_embedding_incorrect.shape)
    return X_embedding,X_embedding_incorrect

if __name__ == "__main__":
	with open('data/sample_data/result/column_names.txt', 'r') as file:
	    saved_columns = [line.strip() for line in file.readlines()]

	current_directory = os.getcwd()
	# print("Current directory:", current_directory)
	directory_name = current_directory + "/data/sample_data"
	# print("directory:",directory_name)
	filename_list,label_list = get_samples_and_labels(directory_name)
	k = int(input("Please enter the value of k："))
	target_name = directory_name + "/result/sample_data_" + str(k) + "mer.tsv"
	get_kmers(k,directory_name,target_name)

	X_NULL = pd.DataFrame({'name': filename_list})
	X = pd.read_csv(target_name,sep = "\t",header=None)
	print("X:",X)
	y = pd.DataFrame(label_list)
	y = y.astype(int)
	print("y",y,y.shape)
	X_embedding,X_embedding_incorrect = get_embedding("data/sample_data/test/result/merged_correct_sequence_embeddings_1024.tsv","data/sample_data/test/result/merged_incorrect_sequence_embeddings_1024.tsv")

	#Basic processing of sequences.
	X.fillna(0,inplace=True)
	X = X.T
	X.columns = X.iloc[0,:]
	X = X.iloc[1:]
	X.columns.values[0] = 'name'

	X_kmer = pd.merge(X_NULL, X, on='name', how='left')
	X = X_kmer
	print("X:",X)

	X_name = X.iloc[:,0]
	X = X.iloc[:,1:].astype(float)
	X = X[saved_columns]
	print("After removing rare k-mers, the number of features is::",len(X.columns))

	#Convert frequency counts to frequencies.
	X_sum = X.sum(axis = 1)
	X_new = (X.T / X_sum).T
	X = X_new
	X.fillna(0,inplace=True)
	X = pd.concat([X_name, X], axis=1)
	print("X:",X)

	print("X_embedding",X_embedding)
	merged_data = pd.merge(X, X_embedding, on='name', how='left')
	merged_data_2 = pd.merge(merged_data, X_embedding_incorrect, on='name', how='left')
	X = merged_data_2
	X.iloc[:,-1024:] = -X.iloc[:,-1024:]
	X_columns = X.columns
	X.index = X['name']
	X = X.iloc[:,1:]
	X.fillna(0,inplace=True)
	print("The final feature matrix:",X)

	y = y.values.ravel()  # Check the shape of y_train and reshape it to a one-dimensional array
	X_test = X
	X_test = X_test.values
	y_test = y

	best_model = None
	best_acc = -1
	best_preds = None



	for i in range(1, 6):
	    model = joblib.load(f"model_{k}/model_top_{i}.pkl")
	    y_pred = model.predict(X_test)
	    y_prob = model.predict_proba(X_test)[:, 1]
	    acc = accuracy_score(y_test, y_pred)
	    print(f"The accuracy of the model model_top_{i}.pkl is: {acc:.4f}")

	    if acc > best_acc:
	        best_acc = acc
	        best_model = model
	        best_preds = y_pred

	    print("Test set accuracy: {:.5f}".format(accuracy_score(y_test, y_pred)))
	    print(f'Precision: {precision_score(y_test, y_pred):.5f}')
	    print(f'Recall: {recall_score(y_test, y_pred):.5f}')
	    print(f'AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.5f}')
	    print(f'F1: {f1_score(y_test, y_pred):.5f}')
	    print(f'MCC: {matthews_corrcoef(y_test, y_pred):.5f}')

