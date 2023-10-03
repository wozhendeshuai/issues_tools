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
import evaluate
max_label = 0


def gen():
    # 从数据库加载数据
    # result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]'  limit 500000")
    # # Read and process JSON data
    # json_files = []
    # for issues_db in result:
    #     json_files.append(issues_db[2])
    # 从filter_json_files.json文件加载数据
    json_files = json.loads(open("../data_pre-processing/filter_json_files4.json", 'r').read())
    print(len(json_files)/2)
    # 从json文件中读取数据，每次读取两个文件，然后将两个文件的title和body拼接起来，然后分别进行tokenize，然后将tokenize后的结果转换为id
    for i in range(int(len(json_files)/2), int(len(json_files)), 2):
        # f1 = json_files[i]
        # f2 = json_files[i + 1]
        # json_data1 = json.loads(f1)
        # json_data2 = json.loads(f2)
        if i == len(json_files) - 1:
            break
        json_data1 = json_files[i]
        json_data2 = json_files[i + 1]
        text1 = json_data1["title"] + " " + json_data1["body"]
        text2 = json_data2["title"] + " " + json_data2["body"]

        # Calculate label count based on common labels
        common_labels = set(json_data1["label"]) & set(json_data2["label"])
        label_count = len(common_labels)
        # Convert label_count to a PyTorch tensor
        label_count = torch.tensor(label_count)
        yield {"text1": text1, "text2": text2, "label_count": label_count}


# 生成Dataset后，给max_label赋值
ds = Dataset.from_generator(gen)
max_label = max(ds["label_count"])
print(len(ds))
# 拆分ds为train_ds和test_ds
raw_datasets = ds.train_test_split(train_size=0.8, test_size=0.2, shuffle=True)
# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# tokenized_dataset = tokenizer(
#     raw_datasets['train']["text1"],
#     raw_datasets['train']["text2"],
#     padding=True,
#     truncation=True,
# )


def tokenize_function(example):
    return tokenizer(example["text1"], example["text2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["text1", "text2"])
tokenized_datasets = tokenized_datasets.rename_column("label_count", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=16, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=16, collate_fn=data_collator
)
# samples=tokenized_datasets["train"][:len(tokenized_datasets["train"])]
# batch = data_collator(samples)
# training_args = TrainingArguments("test_trainer")
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=max_label + 1)

optimizer = AdamW(model.parameters(), lr=3e-5)
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model,
                                                                          optimizer)

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
        loss = F.cross_entropy(logits, labels)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_postfix({'loss': loss.item()})
        progress_bar.update(1)



# metric = evaluate.load("accuracy")
# ...
model.eval()  # Set the model to evaluation mode
eval_loss = 0.0
eval_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in eval_dataloader:
        labels = torch.tensor(batch["labels"])
        # labels = batch["labels"].to(device)  # Make sure labels are on the same device

        outputs = model(**batch)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels)
        eval_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        eval_correct += (predictions == labels).sum().item()
        total_samples += len(labels)

average_eval_loss = eval_loss / len(eval_dataloader)
accuracy = eval_correct / total_samples

print(f"Average Evaluation Loss: {average_eval_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

tokenizer.save_pretrained('sentence_relation_pretrained_bert_model_400w')
model.save_pretrained('sentence_relation_pretrained_bert_model_400w')

