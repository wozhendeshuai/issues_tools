import json

from spider.sql_thread import execute_query, execute_select_query
import re
import emoji
import tqdm

# todo: 后期补充过滤方法
def filter_str(desstr, restr=' '):
    # 过滤除中英文、数字、常用符号以外的其他字符

    # desstr为要过滤的字符串,restr为要保留的字符
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^!@#$%^&*()_+-=<>?.,;:|{}[]~]")
    res = res.sub(restr, desstr)
    res = emoji.replace_emoji(res, restr)
    # 将res中所有两个以上的空格替换为一个空格，并将首尾的空格去掉
    # 替换两个以上的空格为一个空格
    res = re.sub(r'\s+', ' ', res)
    # 去掉首尾的空格
    res = res.strip()
    res = res.lower()
    return res


result = execute_select_query(f"SELECT * FROM All_issues where labels not like '[]'")
# Read and process JSON data
json_files = []
for issues_db in result:
    ftemp = json.loads(issues_db[2])
    ftemp["project"] = issues_db[1]
    json_files.append(ftemp)
# 过滤无法预测的标签
bad_labels = ['invalid', 'valid', 'stale', 'version', 'activity', 'triage',
              'good first issue', 'priority', 'wontfix', 'p0', 'p1', 'p2', 'p3', 'p4', 'status', 'resolved',
              'closed', 'pri', 'critical', 'external', 'reply', 'outdate', 'v0', 'v1', 'v2', 'v3', 'v4', 'branch',
              'done', 'approve', 'accept', 'confirm', 'block', 'duplicate', '1.', '2.', '3.', '4.', '5.', '6.', '7.',
              '8.', '9.', '0.', 'release', 'easy', 'hard',
              'archive', 'fix', 'lock', 'regression', 'assign', 'verified', 'medium', 'high', 'affect', 'star', 'ing',
              'progress']
# 过滤特殊字符、过滤无法预测的标签、过滤小于50的标签
print("json_files len: ", len(json_files))
# 过滤无法预测的标签
# 从json文件中读取数据，每次读取文件的title和body拼接起来，进行tokenize，然后将tokenize后的结果转换为id
filter_json_files = []
label_set = set()
project_label_dict = {}
label_count_dict = {}
# 第一次过滤
for i in tqdm.tqdm(range(0, len(json_files)), desc="第一次过滤"):
    json_data1 = json_files[i]
    # json_data1 = json.loads(f1)
    label_data = json_data1["label"]
    project_str = json_data1["project"]
    if not project_label_dict.__contains__(project_str):
        project_label_dict[project_str] = set()
    # Calculate label count based on common labels
    common_labels = []
    issues_labels = []
    for key in label_data:
        key = key.lower()
        # 首先过滤特殊字符 如果key包含/u的字符，说明是unicode编码，需要去除其中部分的Unicode字符并转换为小写
        key = filter_str(key)
        if key.__len__() == 0:
            continue
        else:
            # 过滤无法预测的标签
            is_bad_label = False
            for each_bad_label in bad_labels:
                if each_bad_label in key:
                    is_bad_label = True
                    break
        if is_bad_label:
            continue
        else:
            label_set.add(key)
            issues_labels.append(key)
            project_label_dict[project_str].add(key)
            if label_count_dict.__contains__(key):
                label_count_dict[key] = label_count_dict[key] + 1
            else:
                label_count_dict[key] = 1
    if issues_labels.__len__() == 0:
        continue
    json_data1["label"] = issues_labels
    filter_json_files.append(json_data1)
# 过滤小于50的标签
less_50_labels = []
for label_str in label_count_dict.keys():
    if label_count_dict[label_str] < 50:
        label_set.remove(label_str)
        less_50_labels.append(label_str)
        for project_str in project_label_dict.keys():
            if project_label_dict[project_str].__contains__(label_str):
                project_label_dict[project_str].remove(label_str)
for less_50_label in less_50_labels:
    label_count_dict.pop(less_50_label)
# 第二次过滤
filter_json_files2 = []
for i in tqdm.tqdm(range(0, len(filter_json_files)), desc="第二次过滤"):
    json_data1 = filter_json_files[i]
    label_data = json_data1["label"]
    # 如果label_data中的标签在less_50_labels中，那么就将其删除
    for label_str in label_data:
        if less_50_labels.__contains__(label_str):
            label_data.remove(label_str)
    if label_data.__len__() == 0:
        continue
    else:
        json_data1["label"] = label_data
        filter_json_files2.append(json_data1)
print("filter_json_files len: ", len(filter_json_files))
print("label_set len: ", len(label_set))
# print("label_set: ", label_set)
print("project_label_dict len: ", len(project_label_dict))
print("filter_json_files2 len: ", len(filter_json_files2))
print("label_count_dict len: ", len(label_count_dict))
# print("project_label_dict: ", project_label_dict)
for key in project_label_dict.keys():
    temp = project_label_dict[key]
    project_label_dict[key] = list(temp)
# print('project_label_json: ', json.dumps(project_label_dict))
# 将Project_label_dict写入文件
with open('project_label_dict.json', 'w') as f:
    json.dump(project_label_dict, f)
# 将label_count_dict按照value值从大到小排列
label_count_dict = sorted(label_count_dict.items(), key=lambda x: x[1], reverse=True)
label_count_file = {}
for key in label_count_dict:
    label_count_file[key[0]] = key[1]
with open('label_count_file.json', 'w') as f:
    json.dump(label_count_file, f)


