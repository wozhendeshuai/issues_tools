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

# 此文件的主要工作是收集表中所有的labels，并将其保存到label_file.json文件中
max_label = 0

result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]' ")
label_num = {}
# Read and process JSON data
label_file = {}
for issues_db in result:
    json_data = json.loads(issues_db[2])
    if label_num.__contains__(len(json_data["label"])):
        label_num[len(json_data["label"])] = label_num[len(json_data["label"])] + 1
    else:
        label_num[len(json_data["label"])] = 1
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
print(label_num)
import matplotlib.pyplot as plt
# 绘制折线图x,y轴的名称
# 为label_num绘制图像 其中x是label_num的key，y是label_num的value
x = list(label_num.keys())
y = list(label_num.values())
plt.xlabel("label_num")
plt.ylabel("label_count")
# 绘制折线图


# 绘制折线图，带有圆点标记
plt.scatter(x, y )

# 显示图形
plt.show()
