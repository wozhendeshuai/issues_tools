import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from spider.sql_thread import execute_query, execute_select_query

result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]' limit 1000")
# Read and process JSON data
data = []
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
    a = {"text1": text1, "text2": text2, "label_count": label_count}
    # 值传递a
    b = a.copy()
    data = data.append(b)


# Define Dataset class
class SimilarityDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_label = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text1 = item["text1"]
        text2 = item["text2"]
        labels = item["label_count"]  # Assuming label_count is an integer

        encoding = self.tokenizer(text1, text2, return_tensors='pt', truncation=True, padding='max_length')
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        self.max_label = max(self.max_label, labels)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Create dataset and data loader
dataset = SimilarityDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load tokenizer and model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=dataset.max_label)
# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3

# Training loop
model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(torch.device("mps"))
        attention_mask = batch['attention_mask'].to(torch.device("mps"))
        labels = batch['labels'].to(torch.device("mps"))

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': loss.item()})

# Save the pretrained model
# model.save_pretrained('pretrained_bert_model')
