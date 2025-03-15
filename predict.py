import os
import time
import tqdm
import pandas as pd
from functools import reduce
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from xgboost import XGBClassifier


#Obtain the sample names and their corresponding labels.
def get_samples_and_labels(directory_name):
    directory = directory_name + "/data"
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
    print("filename_list:",filename_list)
    print("label_list:",label_list)
    return filename_list,label_list

#Calculate the k-mers for each sample and aggregate them into a matrix.
def get_kmers(k,directory_name,target_name):
    directory = directory_name + "/data"
    # Calculate the k-mers
    mkdir_command = f"mkdir data/sample_data/data/kmer/{k}mer"
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


current_directory = os.getcwd()
# print("Current directory:", current_directory)
directory_name = current_directory + "/data/sample_data"
# print("directory:",directory_name)
filename_list,label_list = get_samples_and_labels(directory_name)
k = int(input("Please enter the value of k："))
target_name = directory_name + "/result/sample_data_" + str(k) + "mer.tsv"
get_kmers(k,directory_name,target_name)

X = pd.read_csv(target_name,sep = "\t",header=None)
y = pd.DataFrame(label_list)
print("y",y,y.shape)
X_embedding,X_embedding_incorrect = get_embedding("data/sample_data/result/merged_correct_sequence_embeddings_1024.tsv","data/sample_data/result/merged_incorrect_sequence_embeddings_1024.tsv")

#Basic processing of sequences.
X.fillna(0,inplace=True)
X = X.T
X.columns = X.iloc[0,:]
X = X.iloc[1:]
X = X.iloc[:,1:].astype(float)
print("X:",X)

# Convert the frequency count to relative frequency.
mean_values = X.mean()
sorted_mean_values = sorted(mean_values)
index_mean_value = sorted_mean_values[int((len(sorted_mean_values)/5)*4)]
# print("index_mean_value:",index_mean_value)
X_1 = X
X_1 = X_1.drop(columns=mean_values[mean_values < index_mean_value].index)
X = X_1
print("After removing rare k-mers, the number of features is::",len(X.columns))

#Convert frequency counts to frequencies.
X_sum = X.sum(axis = 1) #serise结构
# print("X_sum:",X_sum)
X_new = (X.T / X_sum).T
X = X_new
X.fillna(0,inplace=True)
# print("X:",X)

#Concatenate `X`, `X_embedding`, and `X_embedding_incorrect` by index.
X_embedding["name"] = X.index
X_embedding_incorrect["name"] = X.index
X_embedding = X_embedding.set_index("name")
merged_data = pd.concat([X, X_embedding], axis=1)
X_embedding_incorrect = X_embedding_incorrect.set_index("name")
merged_data_2 = pd.concat([merged_data, X_embedding_incorrect], axis=1)
print("The final feature matrix:",merged_data_2)
X = merged_data_2
X.iloc[:,-1024:] = -X.iloc[:,-1024:]
X_columns = X.columns
index = X.index

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
X.index = index
print("The normalized X:",X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print("X_train:",X_train,"X_test:",X_test,"y_train:",y_train,"y_test",y_test)

#The pipeline object
pipeline = Pipeline([
    ('feature_selection', 'passthrough'),
    ('classifier', 'passthrough')
])

#Grid Search Optimization
param_grid = [
    {
        'feature_selection': [VarianceThreshold()],
        'feature_selection__threshold': [0.01, 0.05, 0.1, 0.2],  # VarianceThreshold 的参数
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    {
        'feature_selection': [SelectKBest(f_classif)],
        'feature_selection__k': [10, 50, 100],  
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    {
        'feature_selection': [PCA()],
        'feature_selection__n_components': [50, 100],
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    {
        'feature_selection': [VarianceThreshold()],
        'feature_selection__threshold': [0.01, 0.05, 0.1, 0.2],
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'feature_selection': [SelectKBest(f_classif)],
        'feature_selection__k': [10, 50, 100],
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['linear', 'rbf']
    }, {
        'feature_selection': [PCA()],
        'feature_selection__n_components': [50, 100],
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'feature_selection': [VarianceThreshold()],
        'feature_selection__threshold': [0.01, 0.05, 0.1, 0.2],
        'classifier': [XGBClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }, {
        'feature_selection': [SelectKBest(f_classif)],
        'feature_selection__k': [10, 50, 100],
        'classifier': [XGBClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }, {
        'feature_selection': [PCA()],
        'feature_selection__n_components': [50, 100],
        'classifier': [XGBClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
]

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, n_jobs=-1, scoring='accuracy', verbose=2)
y_train = y_train.values.ravel()   # Check the shape of y_train and reshape it to a one-dimensional array
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Using the best parameters for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
prediction = pd.DataFrame(y_pred)
print("y_test:",y_test)
print("y_pred:",y_pred)
prediction.index = y_test.index
# prediction = pd.concat([y_test,y_pred.iloc[:,-1]], axis=1)
# prediction.insert(0, "sample_name", y_test.index)
prediction.columns = ["predicted_label"]
# prediction = prediction[["sample_name", "real_label","predicted_label"]]
prediction.to_csv("data/sample_data/result/prediction.tsv",sep = "\t")
print("The prediction results have been written to the file：data/sample_data/result/prediction.tsv")

# Accuracy
print("Test set accuracy: {:.5f}".format(accuracy_score(y_test, y_pred)))
# Precision
precision = precision_score(y_test.astype(int), y_pred.astype(int))
print(f'Precision: {precision:.5f}')
# Recall
recall = recall_score(y_test.astype(int), y_pred.astype(int))
print(f'Recall: {recall:.5f}')
# AUC
auc_score = roc_auc_score(y_test.astype(int), y_pred.astype(int))
print(f'AUC: {auc_score:.5f}')
f1 = f1_score(y_test.astype(int), y_pred.astype(int))
mcc = matthews_corrcoef(y_test.astype(int), y_pred.astype(int))
print(f'f1: {f1:.5f}')
print(f'mcc: {mcc:.5f}')
