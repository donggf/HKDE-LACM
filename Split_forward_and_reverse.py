import os
import pandas as pd

# Aggregate individual sequence files.
def split_forward_and_reverse(directory_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Categorize the files.
    train_files = sorted([f for f in files if f.startswith("train")], key=lambda x: int(x.split(".")[0].split("_")[-1]))
    val_files = sorted([f for f in files if f.startswith("val")], key=lambda x: int(x.split(".")[0].split("_")[-1]))
    test_files = sorted([f for f in files if f.startswith("test")], key=lambda x: int(x.split(".")[0].split("_")[-1]))
    # print("train_files: ", train_files)

    # Merge the contents of the files.
    def merge_files(file_list, prefix):
        combined_data = pd.DataFrame()
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)
            
            file1 = output_directory + prefix +".tsv"
            file2 = combined_data
            
        df1 = pd.read_csv(file1, sep='\t').iloc[0:,0]
        df2 = combined_data
        df3 = pd.read_csv(file1, sep='\t').iloc[0:,-1]

        merged_df = pd.concat([df1, df2], axis=1)
        merged_df = pd.concat([merged_df, df3], axis=1)
        merged_df.columns = ['name'] + [f"feature_{i}" for i in range(1, 769)] + ['prediction', 'label']
        print(prefix + "_correct_predictions.csv：",merged_df)
            
        correct_predictions = merged_df[merged_df['prediction'] == merged_df['label']]
        incorrect_predictions = merged_df[merged_df['prediction'] != merged_df['label']]
        correct_file = output_directory + prefix + "_correct_predictions.csv"
        incorrect_file = output_directory + prefix + "_incorrect_predictions.csv"
        correct_predictions.to_csv(correct_file, index=False)
        incorrect_predictions.to_csv(incorrect_file, index=False)
        print(f"The rows with correct predictions have been saved to {correct_file}")
        print(f"The rows with incorrect predictions have been saved to {incorrect_file}")
        
    # Merge the train, val, and test files.
    merge_files(train_files, "train")
    merge_files(val_files, "val")
    merge_files(test_files, "test")

directory_path = "data/sample_data/embedding_vector"
output_directory = "data/sample_data/result/"
split_forward_and_reverse(directory_path, output_directory)
