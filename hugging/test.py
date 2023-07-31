import threading
from sklearn.cluster import KMeans, DBSCAN, Birch
import requests
from utils.access_token import get_token
import time
import json
from spider.sql_thread import execute_query, execute_select_query
from concurrent.futures import ThreadPoolExecutor
import re
# 添加进度条
import tqdm
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer


class issues:
    def __init__(self, owner_repo_name, number, title, created_at, user, body, label):
        self.owner_repo_name = owner_repo_name  # owner_name + "_" + repo_name
        self.number = number
        self.title = title
        self.created_at = created_at
        self.user = user  # str(issues_user_id) + '-' + issues_user_name
        self.body = body
        self.label = label

    def print(self):
        print("owner_repo_name:" + self.owner_repo_name, "number:" + self.number, "title:" + self.title,
              "created_at:" + self.created_at, "user:" + self.user, "body:" + self.body, "label:" + self.label)

    def __str__(self):
        return "owner_repo_name:" + self.owner_repo_name + " number:" + self.number + " title:" + self.title + " created_at:" + self.created_at + " user:" + self.user + " body:" + self.body + " label:" + self.label

    def __repr__(self):
        return "owner_repo_name:" + str(self.owner_repo_name) + " number:" + str(
            self.number) + " title:" + self.title + " created_at:" + self.created_at + " user:" + self.user + " body:" + self.body + " label:" + self.label


def get_all_labels(train_dataset, field='labels'):
    labels = set()
    for each_data in train_dataset:
        for each_label in each_data.label:  # not project_labels
            labels.add(each_label)
    return list(labels)


def output_cluster(filename, num_clusters, clustering_model, labels):
    with open(filename, 'w', encoding='utf-8') as outf:
        collect_labels = []
        for i in range(num_clusters):
            collect_labels.append([])

        for i, each_cluster in enumerate(clustering_model.labels_):
            collect_labels[each_cluster].append(i)

        for each_cluster, each_sample_array in enumerate(collect_labels):
            outf.write(f'Cluster {each_cluster} :\n')
            for each_sample in each_sample_array:
                outf.write(f'{labels[each_sample]}\n')

            outf.write('\n')


def pre_data():
    # 过滤无法预测的标签
    bad_labels = ['invalid', 'valid', 'stale', 'version', 'activity', 'triage',
                  'good first issue', 'priority', 'wontfix', 'p0', 'p1', 'p2', 'p3', 'p4', 'status', 'resolved',
                  'closed', 'pri', 'critical', 'external', 'reply', 'outdate', 'v0', 'v1', 'v2', 'v3', 'v4', 'branch',
                  'done', 'approve', 'accept', 'confirm', 'block', 'duplicate', '1.', '0.', 'release', 'easy', 'hard',
                  'archive', 'fix', 'lock', 'regression', 'assign', 'verified', 'medium', 'high', 'affect', 'star',
                  'progress']

    result = execute_select_query(
        f"select * from All_issues where labels not like '[]'")
    issues_list = []
    # 封装issues对象并统计每个label数量
    label_counter = {}
    filter_left_labels = set()
    # 根据count_threshold过滤label
    count_threshold = 50
    for row in tqdm.tqdm(iterable=result):
        # print(row[0])
        issues_json = json.loads(row[2])
        # print(issues_json)
        temp_issues = issues(row[1], issues_json['number'], issues_json['title'], issues_json['created_at'],
                             issues_json['user'], issues_json['body'], issues_json['label'])
        issues_list.append(temp_issues)
        issues_labels = issues_json['label']
        for each_label in issues_labels:
            if each_label not in label_counter:
                label_counter[each_label] = 0
            label_counter[each_label] += 1

    for each_label, count in label_counter.items():
        if label_counter[each_label] > count_threshold:
            filter_left_labels.add(each_label)
    # 打印所有label的数量
    print(f'label_counter: {len(label_counter)}')
    # 打印过滤后的label数量
    print(f'filter left labels: {len(filter_left_labels)}')
    print(len(issues_list))

    # 预处理后的数据
    preprocess_dataset = []
    # 过滤低频标签和bad label
    for each_issue in tqdm.tqdm(iterable=issues_list):
        labels = []
        for each_label in each_issue.label:
            is_good = True
            for each_bad_label in bad_labels:
                if each_bad_label in each_label:
                    is_good = False
                    break
            # 过滤低频标签
            if is_good and each_label in label_counter and label_counter[each_label] > count_threshold:
                labels.append(each_label)
        if len(labels) > 0:
            each_issue.label = labels
            preprocess_dataset.append(each_issue)
    print(len(preprocess_dataset))
    return preprocess_dataset


def pre_cluster(params, preprocess_dataset):
    # 十折交叉验证
    kf = KFold(n_splits=params['n_fold'], shuffle=True, random_state=params['seed'])
    for it, (train_index, test_index) in enumerate(kf.split(preprocess_dataset)):
        if params['no_fold'] and it > 0:
            break
        print('train_index num:', len(train_index), 'test_index num:', len(test_index))
        train_dataset = [preprocess_dataset[i] for i in train_index]
        test_dataset = [preprocess_dataset[i] for i in test_index]

        # 收集所有标签
        all_labels = get_all_labels(train_dataset)
        if params['direct'] == False:
            # 标签聚类
            if params['device'] < 0:
                embedding_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens', device='cpu')
            else:
                embedding_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens',
                                                      device='mps')
            if params['cluster_method'] == 'kmeans':
                clustering_model = KMeans(n_clusters=params['num_clusters'], random_state=params['seed'])
            elif params['cluster_method'] == 'dbscan':
                clustering_model = DBSCAN(eps=params['dbscan_eps'], min_samples=params['dbscan_min_samples'])
            elif params['cluster_method'] == 'birch':
                clustering_model = Birch(threshold=params['birch_threshold'],
                                         branching_factor=params['birch_branching_factor'])

            label_embeddings = embedding_model.encode(all_labels)
            clustering_model.fit(label_embeddings)

            print('write to cluster file.')
            output_cluster("train_and_test_cluster" + str(params['num_clusters']) + ".txt", params['num_clusters'],
                           clustering_model, all_labels)

            # # 构建embedding数据集
            # embedding_train_dataset = embedding_dataset(train_dataset, embedding_model, clustering_model)
            # embedding_test_dataset = embedding_dataset(test_dataset, embedding_model, clustering_model)


if __name__ == "__main__":
    result = []

    n_cluster_list = [2, 4, 8, 16, 32, 64, 128, 256]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_list = [0.00005, 0.0005, 0.005, 0.05, 0.5]
    k_top_abstract_list = [2, 4, 8, 16]
    params = []
    for n_cluster in n_cluster_list:
        for threshold in threshold_list:
            param = {
                'device': 1,
                'n_experiment': 1,
                'model_type': 'cnn',
                'seed': 0,
                'cluster_method': 'kmeans',
                'num_clusters': n_cluster,
                'top_k': 2,
                'n_fold': 3,
                'direct': False,
                'classic': False,
                'word_embed': None,
                'embed_size': 50,
                'result_dir': 'param_search_result',
                'threshold': threshold,
                'no_fold': True,
                'test_batch': 256
            }
            params.append(param)
    preprocess_dataset = pre_data()
    for each_param in tqdm.tqdm(iterable=params):
        print(each_param)
        each_result = pre_cluster(each_param, preprocess_dataset)
        result.append(
            {
                'n_cluster': each_param['num_clusters'],
                'k_top': each_param['top_k'],
                'threshold': each_param['threshold'],
                'metrics': each_result
            }
        )

        with open('param_search.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(result))
