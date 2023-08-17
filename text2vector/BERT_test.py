import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, \
    BertForNextSentencePrediction, AdamW, DataCollatorWithPadding, get_scheduler
from spider.sql_thread import execute_query, execute_select_query
import tqdm
from datasets import Dataset
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

max_label = 0


def gen():
    result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]'  limit 100")
    # Read and process JSON data
    json_files = []
    for issues_db in result:
        json_files.append(issues_db[2])
    # 从json文件中读取数据，每次读取两个文件，然后将两个文件的title和body拼接起来，然后分别进行tokenize，然后将tokenize后的结果转换为id
    for i in range(0, len(json_files), 2):
        f1 = json_files[i]
        f2 = json_files[i + 1]
        json_data1 = json.loads(f1)
        json_data2 = json.loads(f2)

        text1 = json_data1["title"] + " " + json_data1["body"]
        text2 = json_data2["title"] + " " + json_data2["body"]

        # Calculate label count based on common labels
        common_labels = set(json_data1["label"]) & set(json_data2["label"])
        label_count = len(common_labels)
        # Convert label_count to a PyTorch tensor
        label_count = torch.tensor(label_count)
        yield {"row_id": i, "text1": text1, "text2": text2, "label_count": label_count}


# 生成Dataset后，给max_label赋值
ds = Dataset.from_generator(gen)
max_label = max(ds["label_count"])
# 拆分ds为train_ds和test_ds
raw_datasets = ds.train_test_split(train_size=0.8, test_size=0.2, shuffle=True)

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence_relation_pretrained_bert_model')


def tokenize_function(example):
    return tokenizer(example["text1"], example["text2"], truncation=True)


# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets=ds.map(tokenize_function)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["text1", "text2", "row_id"])
tokenized_datasets = tokenized_datasets.rename_column("label_count", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets, shuffle=True, batch_size=16, collate_fn=data_collator
)

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained('sentence_relation_pretrained_bert_model',
                                                           num_labels=max_label + 1, ignore_mismatched_sizes=True)

optimizer = AdamW(model.parameters(), lr=3e-5)
train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model,optimizer)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
device = torch.device("mps")
model.to(device)

progress_bar = tqdm.tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        # batch = {k: v.to(device) for k, v in batch.items()}
        labels = torch.tensor(batch["labels"])
        # labels = batch["labels"].to(device)  # Make sure labels are on the same device
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        print("outputs:{}", outputs)
        print("logits:{}", logits)
        print("predictions:{}", predictions)
        print("true_labels:{}", labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_postfix({'loss': loss.item()})
        progress_bar.update(1)

# model.save_pretrained('sentence_issues_pretrained_bert_model')
