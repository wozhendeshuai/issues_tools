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

result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]' ")
# Read and process JSON data
label_file = {}
for issues_db in result:
    json_data = json.loads(issues_db[2])
    for label in json_data["label"]:
        # 将label全部转换为小写
        label = label.lower()
        if label_file.__contains__(label):
            label_file[label] = label_file[label] + 1
        else:
            label_file[label] = 1
# label_file 按照计数从大到小排列一下
# label_file = sorted(label_file.items(), key=lambda x: x[1], reverse=True)
label_sort = sorted(label_file.items(), key=lambda x: x[1], reverse=True)
label_dict = {}
for key in label_sort:
    label_dict[key[0]] = key[1]
# 将label_file写入文件，并按照value值从大到小排列
with open("label_file.json", "w") as f:
    json.dump(label_dict, f)
