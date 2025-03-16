from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os
import numpy as np

# Load the preprocessed data.
data_files = {
    "train": "data/sample_data/processed/train.tsv",
    "validation": "data/sample_data/processed/val.tsv",
    "test": "data/sample_data/processed/test.tsv",
}
dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# Load the pre-trained DNABERT model and tokenizer.
tokenizer = BertTokenizer.from_pretrained("DNABERT-6")
model = BertForSequenceClassification.from_pretrained("DNABERT-6", num_labels=2)

torch.cuda.empty_cache()

# Data encoding for DNABERT.
def encode(examples):
    return tokenizer(examples["sequence"], truncation=True, padding="max_length", max_length=512)

encoded_dataset = dataset.map(encode, batched=True)

# Create a custom Trainer class to extract embeddings (sequence_embedding).
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions = []
        self.embeddings = []

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):  ## 添加 **kwargs 来忽略不期望的参数
        # 获取模型的输出，包括隐藏状态
        outputs = model(**inputs, output_hidden_states=True)
        loss = outputs.loss
        logits = outputs.logits  #这是模型对每个输入样本的原始预测输出
        hidden_states = outputs.hidden_states   #每一层的隐藏状态

        if return_outputs:
            return loss, outputs
        return loss

# Fine-tune the model.
training_args = TrainingArguments(
    output_dir="data/sample_data/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="data/sample_data/logs",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
)
trainer.train()

trainer.save_model("data/sample_data/finetuned_DNABERT6")  # Save the fine-tuned model.
print("Training complete!")
