
from sklearn.cluster import KMeans

import json

from transformers import AutoTokenizer

def output_cluster(filename, num_clusters, clustering_model, labels):
    new_dict = {}
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
                new_dict[labels[each_sample]]=each_cluster
            outf.write('\n')
    # 将读入的new_dict写入文件
    with open('label_cluster.json', 'w', encoding='utf-8') as f:
        json.dump(new_dict, f, ensure_ascii=False)

# Load and preprocess the issues data
data = {}
with open('label_file.json', 'r') as f:
    data = json.load(f)

labels = [item for item in data.keys() if data[item] > 50]
# labels_text = [' '.join(label) for label in labels]
labels_text = labels
# TF-IDF vectorization
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(labels_text)
tokenizer = AutoTokenizer.from_pretrained("../text2vector/sentence_relation_pretrained_bert_model/")
X = tokenizer(labels_text, padding=True, truncation=True, return_tensors="pt")
X=X.get('input_ids')
# Elbow Method to determine optimal number of clusters
distortions = []
sil_scores = []
max_clusters = 256  # You can adjust this range based on your data

# for k in tqdm.tqdm(range(1, max_clusters + 1)):
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(X)
#     distortions.append(kmeans.inertia_)
#     if k > 1:
#         sil_scores.append(silhouette_score(X, kmeans.labels_))

# Plotting the Elbow Method curve
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, max_clusters + 1), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.title('Elbow Method')
#
# plt.subplot(1, 2, 2)
# plt.plot(range(2, max_clusters + 1), sil_scores, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score')
#
# plt.tight_layout()
# plt.show()

# Based on the plots, you can visually determine the optimal number of clusters

# Once you have determined the optimal number of clusters, perform KMeans clustering
optimal_clusters = 128  # Adjust based on the Elbow Method plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=2)
kmeans.fit(X)

# Print cluster labels for each issue
output_cluster('cluster_output.txt', optimal_clusters, kmeans, labels)
