import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

final_df = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/final_unified_embeddings_aggregated_100.pkl')
print(final_df)

# # Assuming `unified_embedding` is a list of embeddings (one per compound) for each food
# final_df['num_compounds'] = final_df['unified_embedding'].apply(lambda x: len(x))

# # Inspect the first few rows to check the number of compounds per food
# print(final_df[['food_name', 'num_compounds']])

# # Set an upper threshold for the number of compounds
# upper_threshold = 10000
# lower_threshold = 10

# # Filter foods with number of compounds below the threshold. We are excluding 29 foods with over 10000 compounds and 35 with under 10.
# df_filtered_food = final_df[(final_df['num_compounds'] >= lower_threshold) & (final_df['num_compounds'] <= upper_threshold)]
# print(df_filtered_food.shape)
# print(df_filtered_food)

# df_filtered_food_only = df_filtered_food[['food_name', 'unified_embedding']]

# df_filtered_food_only.to_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/df_filtered_food_50.pkl')

# print(f"Number of foods with compounds between {lower_threshold} and {upper_threshold}: {len(df_filtered_food)}")
# # Determine the 95th percentile for the number of compounds
# max_compounds = int(df_filtered_food['num_compounds'].quantile(0.95))

# print(f"The 95th percentile for the number of compounds is: {max_compounds}")


# final_df['unified_embedding_flat'] = final_df['unified_embedding_padded'].apply(lambda x: x.flatten())

# Convert the list of flattened embeddings into a 2D NumPy array
X_flat = np.array(final_df['unified_embedding'].tolist())
print(X_flat.shape)
# Perform K-means clustering (specify the number of clusters)
kmeans = KMeans(n_clusters=13, random_state=42).fit(X_flat)


# # Add the cluster labels to the DataFrame
final_df['kmeans_cluster'] = kmeans.labels_
print(final_df)
# # Visualize the number of foods in each cluster
final_df['kmeans_cluster'].value_counts().plot(kind='bar')
plt.title('Number of Foods in Each Cluster (kmeans)')
plt.xlabel('Cluster')
plt.ylabel('Number of Foods')
plt.show()

# Print a few foods from each cluster
for cluster in range(5):
    print(f"Foods in Cluster {cluster}:")
    print(final_df[final_df['kmeans_cluster'] == cluster]['food_name'].head())

# save the clusters

## DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=4)  # Adjust 'eps' and 'min_samples' as needed
dbscan.fit(np.array(final_df['unified_embedding'].tolist()))

# Add the cluster labels to the DataFrame
final_df['dbscan_cluster'] = dbscan.labels_

# Print a few examples from each cluster
unique_clusters = set(dbscan.labels_)
for cluster in unique_clusters:
    print(f"Foods in Cluster {cluster}:")
    print(final_df[final_df['dbscan_cluster'] == cluster]['food_name'].head())

final_df['dbscan_cluster'].value_counts().plot(kind='bar')
plt.title('Number of Foods in Each Cluster (dbscan)')
plt.xlabel('Cluster')
plt.ylabel('Number of Foods')
plt.show()

## save df
final_df[['food_name', 'kmeans_cluster', 'dbscan_cluster']].to_csv('C:/Users/labro/Downloads/Thesis_Food/final_unified_embeddings_aggregated_100_clusters.csv')
final_df.to_pickle('C:/Users/labro/Downloads/Thesis_Food/final_unified_embeddings_aggregated_100_clusters.pkl')
## HDBSCAN
# import hdbscan

# # Assuming X is a 2D array after flattening or aggregation
# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# # Create the HDBSCAN model
# # Calculate the pairwise distances using the custom distance function (same as above)
# distance_matrix = pdist(final_df['unified_embedding'].tolist(), metric=lambda u, v: custom_distance(np.array(u), np.array(v)))

# # Convert the distance matrix to a square form
# distance_matrix_square = squareform(distance_matrix)

# hdb = hdbscan.HDBSCAN(metric='precomputed', min_samples=5, min_cluster_size=10)
# final_df['cluster'] = hdb.fit(distance_matrix_square)

# # Add the cluster labels to the DataFrame
# # final_df['cluster'] = hdb.labels_

# # Print a few examples from each cluster
# unique_clusters = set(hdb.labels_)
# for cluster in unique_clusters:
#     print(f"Foods in Cluster {cluster}:")
#     print(final_df[final_df['cluster'] == cluster]['food_name'].head())



def PCA_plot(data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Assuming `data` contains your high-dimensional embeddings (e.g., 400 dimensions)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=None, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(label='Cluster Label')
    plt.title("2D PCA Projection")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


def Umap_plot(data):
    import umap

    # Reduce dimensions to 2D with UMAP
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_2d.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=None, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(label='Cluster Label')
    plt.title("2D UMAP Projection")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.show()


# PCA_plot(X_flat)
# Umap_plot(X_flat)

## Umap + HDBSCAN
import umap
import hdbscan
# Step 1: Apply UMAP to reduce dimensions (e.g., from 400 to 10 for clustering)
umap_model = umap.UMAP(n_components=10, random_state=42)
umap_embeddings = umap_model.fit_transform(X_flat)

# Step 2: Apply HDBSCAN for clustering on the UMAP-reduced data
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
hdbscan_labels = hdbscan_model.fit_predict(umap_embeddings)

# Add cluster labels to your dataframe
final_df['hdbscan_cluster_label'] = hdbscan_labels
final_df['hdbscan_cluster_label'].value_counts().plot(kind='bar')
plt.title('Number of Foods in Each Cluster (hdbscan)')
plt.xlabel('Cluster')
plt.ylabel('Number of Foods')
plt.show()
