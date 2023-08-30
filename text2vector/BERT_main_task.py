import os
import json
import torch
from torch.utils._contextlib import F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, \
    BertForNextSentencePrediction, AdamW, DataCollatorWithPadding, get_scheduler
from spider.sql_thread import execute_query, execute_select_query
import tqdm
from datasets import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import pandas as pd
import numpy as np

# 在此文件中加载预训练模型，然后使用预训练模型在主任务（单文本issue文本的多分类任务）进行训练
max_label = 128  # 这个值应该是聚类的簇数


#  order by rand()
def gen():
    result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]' order by rand() limit 20000")
    # 从label_cluster.json文件中读取label_cluster
    label_cluster = json.loads(open("../issues_label/label_cluster.json", "r").read())
    # Read and process JSON data
    json_files = []
    for issues_db in result:
        json_files.append(issues_db[2])

    # 从json文件中读取数据，每次读取文件的title和body拼接起来，进行tokenize，然后将tokenize后的结果转换为id
    for i in range(0, len(json_files)):
        # labels_binary是一个128维的列表
        labels_binary = []
        for j in range(0, max_label):
            labels_binary.append(0)
        f1 = json_files[i]
        json_data1 = json.loads(f1)
        text1 = json_data1["title"] + " " + json_data1["body"]
        # Calculate label count based on common labels
        common_labels = []
        for key in json_data1["label"]:
            if key in label_cluster.keys():
                common_labels.append(label_cluster[key])
                labels_binary[label_cluster[key]] = 1

        # 打印出该issues的label数量，和labels_binary中为1的元素数量
        print("\n json_data1[label]" + str(json_data1["label"]) + "len(json_data1[label])" +
              str(len(json_data1["label"])) + " sum(labels_binary):" + str(sum(labels_binary)))
        # Convert label_count to a PyTorch tensor
        label_count = torch.tensor(labels_binary)
        yield {"row_id": i, "text1": text1, "labels": label_count}


# 生成Dataset后，给max_label赋值
ds = Dataset.from_generator(gen)

# # 将ds写入到csv文件中
# ds.to_csv("data/test.csv")

# 拆分ds为train_ds和test_ds
raw_datasets = ds.train_test_split(train_size=0.8, test_size=0.2, shuffle=True)

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence_relation_pretrained_bert_model')


def tokenize_function(example):
    return tokenizer(example["text1"], truncation=True)


# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = raw_datasets.map(tokenize_function)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["text1", "row_id"])
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=16, collate_fn=data_collator
)
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained('sentence_relation_pretrained_bert_model',
                                                           # 'bert-base-uncased',
                                                           num_labels=max_label,
                                                           ignore_mismatched_sizes=True,
                                                           problem_type="multi_label_classification")

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

# Define a list to store prediction results
all_train_predictions = []
all_train_labels = []

progress_bar = tqdm.tqdm(range(num_training_steps))

model.train()
criterion = BCEWithLogitsLoss()
for epoch in range(num_epochs):
    progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        labels = torch.tensor(batch.pop("labels")).to(device)
        inputs = batch
        inputs.to(device)
        # batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        # print(outputs)

        logits = outputs.logits
        # Apply sigmoid activation to logits
        predictions = torch.sigmoid(logits)
        # Move predictions to CPU and convert to NumPy array
        predictions = predictions.cpu().detach().numpy()

        # Collect predictions and true labels for each sample in the batch
        for i in range(len(predictions)):
            sample_predictions = predictions[i]
            sample_labels = labels[i].cpu().numpy()
            # print(sample_predictions)
            # print(sample_labels)
            all_train_predictions.append(sample_predictions)
            all_train_labels.append(sample_labels)

        loss = criterion(logits, labels.float())  # Use BCEWithLogitsLoss

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_postfix({'loss': loss.item()})
        progress_bar.update(1)

# Convert the list of predictions and labels to NumPy arrays
all_train_predictions = np.array(all_train_predictions)
all_train_labels = np.array(all_train_labels)

# Calculate the number of positive labels for each sample
num_pos_labels = np.sum(all_train_labels, axis=1)
# print(num_pos_labels)
# num_pos_labels，对all_predictions进行排序，取出前num_pos_labels个值
# top_n_indices = np.argsort(all_predictions, axis=1)[:, -num_pos_labels[0]:]
# Get the indices of sorted predictions for each sample
sorted_indices = np.argsort(all_train_predictions, axis=1)[:, ::-1]
# print(sorted_indices)
# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=["Top_N_Predicted_Positions", "True_Label_Positions"])

# Populate the DataFrame with the results
for i in range(len(sorted_indices)):
    true_label_positions = np.where(all_train_labels[i] == 1)[0]  # Get the positions of true labels
    num_pos_labels = len(true_label_positions)  # Count of true positive labels
    top_n_positions = sorted_indices[i, :num_pos_labels]  # Get the top n predicted positions
    true_label_positions_str = " ".join(map(str, true_label_positions))
    results_df.loc[i] = [top_n_positions, true_label_positions_str]

# Save the results DataFrame to a CSV file
results_df.to_csv("data/train_predictions_and_labels_positions.csv", index=False)
# model.save_pretrained('sentence_issues_pretrained_bert_model')
# metric = evaluate.load("accuracy")
# ...
model.eval()  # Set the model to evaluation mode
eval_loss = 0.0
eval_correct = 0
total_samples = 0

# Define a list to store prediction results
all_test_predictions = []
all_test_labels = []
with torch.no_grad():
    progress_bar = tqdm.tqdm(eval_dataloader, desc=f"在测试集的表现")
    for batch in progress_bar:
        labels = torch.tensor(batch.pop("labels")).to(device)
        inputs = batch
        inputs.to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, labels.float())
        eval_loss += loss.item()

        # predictions = torch.argmax(logits, dim=1)
        # eval_correct += (predictions == labels).sum().item()
        # total_samples += len(labels)

        # Apply sigmoid activation to logits
        predictions = torch.sigmoid(logits)
        # Move predictions to CPU and convert to NumPy array
        predictions = predictions.cpu().detach().numpy()

        # Collect predictions and true labels for each sample in the batch
        for i in range(len(predictions)):
            sample_predictions = predictions[i]
            sample_labels = labels[i].cpu().numpy()
            # print(sample_predictions)
            # print(sample_labels)
            all_test_predictions.append(sample_predictions)
            all_test_labels.append(sample_labels)

        loss = criterion(logits, labels.float())  # Use BCEWithLogitsLoss

# 输出测试集中的预测结果和真实结果
# Convert the list of predictions and labels to NumPy arrays
all_test_predictions = np.array(all_test_predictions)
all_test_labels = np.array(all_test_labels)

# Calculate the number of positive labels for each sample
num_pos_labels = np.sum(all_test_labels, axis=1)
# print(num_pos_labels)
# num_pos_labels，对all_predictions进行排序，取出前num_pos_labels个值
# top_n_indices = np.argsort(all_predictions, axis=1)[:, -num_pos_labels[0]:]
# Get the indices of sorted predictions for each sample
sorted_indices = np.argsort(all_test_predictions, axis=1)[:, ::-1]
# print(sorted_indices)
# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=["Top_N_Predicted_Positions", "True_Label_Positions"])

# Populate the DataFrame with the results
for i in range(len(sorted_indices)):
    true_label_positions = np.where(all_test_labels[i] == 1)[0]  # Get the positions of true labels
    if true_label_positions.size == 0:
        total_samples += 1
        eval_correct += 1
        continue
    num_pos_labels = len(true_label_positions)  # Count of true positive labels
    top_n_positions = sorted_indices[i, :num_pos_labels]  # Get the top n predicted positions
    true_label_positions_str = " ".join(map(str, true_label_positions))
    results_df.loc[i] = [top_n_positions, true_label_positions_str]
    # 统计true_label_positions和top_n_positions之间的交集的个数
    total_samples += num_pos_labels
    print(true_label_positions)
    # 计算准确率，计算预测的标签类别top_n_positions和真实的标签类别true_label_positions相同元素的个数，然后除以总的标签类别的个数的
    for j in range(len(top_n_positions)):
        # 判断top_n_positions[j]是否在true_label_positions中
        if top_n_positions[j] in true_label_positions:
            eval_correct += 1
            print(top_n_positions[j])

# Save the results DataFrame to a CSV file
results_df.to_csv("data/test_predictions_and_labels_positions.csv", index=False)

average_eval_loss = eval_loss / len(eval_dataloader)
accuracy = eval_correct / total_samples

print(f"Average Evaluation Loss: {average_eval_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
