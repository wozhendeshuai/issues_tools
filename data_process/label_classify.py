import json

import spider.sql_thread as sql_thread

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
# Get a list of all tables in the database
tables = sql_thread.execute_select_query("SELECT table_name FROM information_schema.tables WHERE  table_schema = 'github_issues_db';")

# Initialize a dictionary to store the count of labels in different tables
label_count = {}
label_set=set()
# Iterate over each table
for table in tables:
    table_name = table[0]
    labels = sql_thread.execute_select_query(f"SELECT labels FROM {table_name};")

    # Count the occurrence of each label in the table
    if table_name not in label_count:
        label_count[table_name] = {}

    for label_str in labels:
        label_list=eval(label_str[0])
        for label in label_list:
            label_set.add(label)
            if label in label_count[table_name]:
                label_count[table_name][label] += 1
            else:
                label_count[table_name][label] = 1

# Print the labels and their count in different tables
for table_name, label_count in label_count.items():
    for label, count in label_count.items():
        print(f"table_name: {table_name} Label: {label}, Count: {count}")
print(label_set)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Vectorize the labels
vectorizer = TfidfVectorizer()
label_vectors = vectorizer.fit_transform(label_set)
label_vectors_plt = np.reshape(label_vectors.toarray(), (label_vectors.shape[0], 2))
#绘制数据分布图
plt.scatter(label_vectors_plt[:, 0], label_vectors_plt[:, 1],  c = "red", marker='o', label='see')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Results')
plt.show()
# Perform clustering analysis
kmeans = KMeans()
kmeans.fit(label_vectors)

# Find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(label_vectors)
    inertia.append(kmeans.inertia_)

# Visualize the results
plt.plot(range(1, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Automatically find the optimal number of clusters
optimal_clusters = None
for i in range(1, len(inertia)-1):
    if inertia[i] - inertia[i-1] > inertia[i+1] - inertia[i]:
        optimal_clusters = i+1
        break

# Perform clustering analysis with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters)
kmeans.fit(label_vectors)
print('write to cluster file.')





output_cluster("train_and_test_cluster.txt", optimal_clusters, kmeans, label_vectors)

# # Visualize the clustering results
# # Assuming you have label_vectors, kmeans.labels_, and plt imported


# Reshape the color_subset array to match the size of label_vectors[:, 0]
color_subset = kmeans.labels_[:label_vectors.shape[0]]
color_subset = np.reshape(color_subset, (label_vectors.shape[0],))
# Example: Convert the sequences to scalar values
label_vectors = np.reshape(label_vectors, (label_vectors.shape[0], 2))
# Plot the scatter plot with the modified color argument
plt.scatter(label_vectors[:, 0], label_vectors[:, 1], c=color_subset)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Results')
plt.show()