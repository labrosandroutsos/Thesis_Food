from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns


df_foodb = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_presence/foodname_compound_presence_0_1.csv', sep=';', index_col=0)

print(df_foodb)
print(df_foodb.shape)

# give me the sum of each row
df_foodb['sum'] = df_foodb.sum(axis=1)
print(df_foodb['sum'])
print(df_foodb.index)
# # Perform hierarchical clustering
Z = linkage(df_foodb, method='average')  # 'ward' method is commonly used, but you can try others like 'single', 'complete', etc.
print(Z)

plt.figure(figsize=(20, 10))
dendrogram(Z, labels=df_foodb.index, leaf_rotation=90, leaf_font_size=8,truncate_mode='lastp', p=40)
# dendrogram(Z, labels=df_foodb.index, leaf_rotation=90, leaf_font_size=8,truncate_mode='level', p=10)
plt.title('Hierarchical Clustering Dendrogram [Average Linkage]')
plt.xlabel('Food')
plt.ylabel('Distance')
plt.savefig('C:/Users/labro/Downloads/Thesis_Food/hierarchical_clustering_dendrogram_average_lastp_p40.png')
plt.show()

clusters = fcluster(Z, criterion='maxclust', t=40)
# Assuming the samples are in a 1D array or list 'sample_names'
cluster_dict = defaultdict(list)

for i, cluster_label in enumerate(clusters):
    cluster_dict[cluster_label].append(df_foodb.index[i])

for cluster, samples in cluster_dict.items():
    print(f"Cluster {cluster}: {samples}")

# Step 5: Write clusters to a file
# with open('C:/Users/labro/Downloads/Thesis_Food/clusters_40_linkage_average.txt', 'w') as f:
#     for cluster, samples in cluster_dict.items():
#         f.write(f"Cluster {cluster}: {', '.join(samples)}\n\n")

# from sklearn.metrics import silhouette_score

# range_n_clusters = list(range(2, 20))
# silhouette_scores = []

# for n_clusters in range_n_clusters:
#     clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
#     cluster_labels = clusterer.fit_predict(df_foodb)
#     silhouette_avg = silhouette_score(df_foodb, cluster_labels)
#     silhouette_scores.append(silhouette_avg)
#     print(f"For n_clusters = {n_clusters}, the average silhouette_score is {silhouette_avg}")

# # Plot silhouette scores
# plt.figure(figsize=(10, 6))
# plt.plot(range_n_clusters, silhouette_scores, marker='o')
# plt.title("Silhouette Scores for Different Numbers of Clusters")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Score")
# plt.savefig('C:/Users/labro/Downloads/Thesis_Food/silhouette_score_clusters.png')
# plt.show()


# agg_cluster = AgglomerativeClustering(distance_threshold=500, n_clusters=None, metric='euclidean', linkage='ward')
# df_foodb['Cluster'] = agg_cluster.fit_predict(df_foodb)
# # Display the number of items in each cluster
# print(df_foodb['Cluster'].value_counts())
# print(df_foodb['Cluster'].head())  # Display the first few rows of the 'Cluster' column
# print(df_foodb['Cluster'].dtype)   # Check the data type of the 'Cluster' column

# # Analyze the distribution of foods across clusters
# # Make sure you reference the 'Cluster' column correctly
# # sns.countplot(x='Cluster', data=df_foodb)
# # plt.title('Number of Foods in Each Cluster')
# # plt.show()

# with open('C:/Users/labro/Downloads/Thesis_Food/cluster_results.txt', 'w') as file:

#     for cluster in range(df_foodb['Cluster'].nunique()):
#         # Write the cluster number
#         file.write(f"Cluster {cluster}:\n")
        
#        # Get the list of items (index) in this cluster and convert to strings
#         items_in_cluster = df_foodb[df_foodb['Cluster'] == cluster].index.tolist()
#         items_in_cluster = [str(item) for item in items_in_cluster]  # Convert each item to a string
        
#         # Write the items in the cluster
#         file.write(", ".join(items_in_cluster) + "\n\n")

# You can also group by clusters to see which foods belong to which cluster
# for cluster in range(n_clusters):
#     print(f"Cluster {cluster}:")
#     print(df[df['Cluster'] == cluster].index.tolist())
#     print("\n")
