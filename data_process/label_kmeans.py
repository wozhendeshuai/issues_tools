from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import json
import utils.data_utils as data_utils

import numpy as np

from sentence_transformers import SentenceTransformer
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
label_set = data_utils.get_label_set()
label_list = list(label_set)
# 标签聚类

embedding_model = SentenceTransformer('D:\\data\\pretrained_model\\all-MiniLM-L6-v2', device=f'cuda:{0}')
clustering_model = KMeans(n_clusters=64, random_state=0)
# if params['cluster_method'] == 'kmeans':
#     clustering_model = KMeans(n_clusters=params['num_clusters'], random_state=params['seed'])
# elif params['cluster_method'] == 'dbscan':
#     clustering_model = DBSCAN(eps=params['dbscan_eps'], min_samples=params['dbscan_min_samples'])
# elif params['cluster_method'] == 'birch':
#     clustering_model = Birch(threshold=params['birch_threshold'], branching_factor=params['birch_branching_factor'])

label_embeddings = embedding_model.encode(label_list)
clustering_model.fit(label_embeddings)

print('write to cluster file.')
output_cluster("train_and_test_cluster.txt", 64, clustering_model, label_list)

# # 构建embedding数据集
# embedding_train_dataset = embedding_dataset(train_dataset, embedding_model, clustering_model)
# embedding_test_dataset = embedding_dataset(test_dataset, embedding_model, clustering_model)

# 平衡数据集
# weights = make_weights_for_balanced_classes(embedding_train_dataset, params['num_clusters'])
