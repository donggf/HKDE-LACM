# 1.  Introduction 
This package implements the "HKDE-LACM: A Hybrid Model for Lactic Acid Bacteria Classification via k-mer and DNABERT-2 Embedding Fusion with Cyclic DE-BO Optimization". The model analyzes the genomic sequences of lactic acid bacteria, leveraging high-dimensional k-mers frequency features and DNABERT-2 embedding technology to achieve classification of lactic acid bacteria.
    
# 2.  Data Sample 
The data can be downloaded from the following website:[http://bioinfor.imu.edu.cn/iprobiotics/public/download.html](http://bioinfor.imu.edu.cn/iprobiotics/public/download.html). Please save the samples into the directory “data\sample_data\data” following the specified format: 
	• Positive samples should be named starting with "A" followed by a numeric identifier (e.g., "A1", "A2"). 
	• Negative samples should be named starting with "B" followed by a numeric identifier (e.g., "B1", "B2").
 Refer to the example samples already provided in the directory for guidance.
    
# 3.  Environment settings 
GPU：NVIDIA RTX 4090D（24GB）× 1; CPU：AMD EPYC 9754（128 core），18 vCPU; Memory：60GB; Operating system：Ubuntu 20.04； PyTorch：2.0.0; Python：3.8; Cuda：11.8
    

### Install Jellyfish on Linux systems:
Refer to the guide:[Jellyfish]([https://blog.csdn.net/qq524730309/article/details/124706296?ops_request_misc=%257B%2522request%255Fid%2522%253A%25226e240537abd6d8251d8a26299c19d968%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=6e240537abd6d8251d8a26299c19d968&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124706296-null-null.142%5Ev102%5Epc_search_result_base5&utm_term=jellyfish&spm=1018.2226.3001.4187](https://github.com/gmarcais/Jellyfish)) 
### Create and activate a virtual environment:

```
python3 -m venv pytorch_env  
source pytorch_env/bin/activate
```

### Install required packages： 

```
python3 -m pip install -r requirements.txt
```

# 4.  Running the Code 
### a) Fine-tune the DNABERT-2 model and obtain the corresponding sequence embeddings
```
bash gain_embeddings.sh
```
### b) Integrate k-mer features with embedding vectors and train the model
```
python3 train.py
```
### a) Evaluate the model on the test set and obtain results
```
python3 test.py
```

# 5.  File Descriptions
Below are the key files and their functionalities:
- `Split_dataset.py`
Split the dataset into training, testing, and validation sets. Each sample is divided into 800-length segments. Processed results are saved to the “data\sample_data\processed” directory.

- `finetune_model.py`
Fine-tune the model using the preprocessed data to obtain a fine-tuned model, which will be saved in the”data\sample_data\finetuned_DNABERT_2” directory.

- `gain_prediction.py`
Use the fine-tuned model to predict genomic sequence segments, generating prediction prediction probabilities and sequence embeddings. Results are saved in the “data\sample_data\embedding_vector”directory

- `Split_forward_and_reverse.py`
Categorize embedding sequences into: 
	• Positive embedding vectors 
 		-Samples with label = 1 and prediction probability > 0.9
  		-Samples with label = 0 and prediction probability < 0.1
	• Negative embedding vectors
		-Samples with label = 1 and prediction probability < 0.1
  		-Samples with label = 0 and prediction probability > 0.9
	Results are saved in the “data\sample_data\result” directory.

- `Obtain_complete_embeddings.py`
- `Obtain_incorrect_complete_embeddings.py`
Compile the embedding vectors of gene sequence fragments into complete positive and negative embedding vectors corresponding to the full gene sequences. Results are saved in the “data\sample_data\result” directory with filenames：merged_correct_sequence_embeddings_1024.tsv和merged_incorrect_sequence_embeddings_1024.tsv.


## Runtime with 57 Positive and 57 Negative Samples 
We provide runtime estimates based on a dataset of 57 positive and 57 negative samples.
### k-mers Calculation Time 
As the k-value increases, the computation time for generating k-mers per sample grows progressively. We currently allocate an 6-second waiting period to fully capture 8-mer sequences from the genomic sequences. Users can adjust the waiting time in the train.py and test.py files according to their chosen k-value.
### Model Fine-tuning 
Approximately 6 hours are required for model fine-tuning.
### Embedding Sequence Extraction 
Generating embedding sequences for all 114 samples takes around 3 hours.
### Prediction with Combined Features 
Runtime varies with k-values: 
• 8-mer: 171 seconds 
• 9-mer: 188 seconds 
• 10-mer: 361 seconds 

