import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster

# Load your data
df_foodb = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_presence/foodname_compound_presence_0_1.csv', sep=';', index_col=0)

# Compute linkage and clusters for chosen parameters
linkage_average = linkage(df_foodb, method='average')
clusters_average_10 = fcluster(linkage_average, criterion='maxclust', t=10)
clusters_average_20 = fcluster(linkage_average, criterion='maxclust', t=20)

# Option 1: PCA Projection
pca = PCA(n_components=2)
pca_projection = pca.fit_transform(df_foodb)

# Plot PCA projections
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=pca_projection[:, 0], y=pca_projection[:, 1], hue=clusters_average_10, palette='viridis', s=50)
plt.title("PCA Projection with 10 Clusters (Average Linkage)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid()

plt.subplot(1, 2, 2)
sns.scatterplot(x=pca_projection[:, 0], y=pca_projection[:, 1], hue=clusters_average_20, palette='viridis', s=50)
plt.title("PCA Projection with 20 Clusters (Average Linkage)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid()
plt.tight_layout()
plt.show()

# Option 2: UMAP Projection
umap_model = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42)
umap_projection = umap_model.fit_transform(df_foodb)

# Plot UMAP projections
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=umap_projection[:, 0], y=umap_projection[:, 1], hue=clusters_average_10, palette='viridis', s=50)
plt.title("UMAP Projection with 10 Clusters (Average Linkage)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid()

plt.subplot(1, 2, 2)
sns.scatterplot(x=umap_projection[:, 0], y=umap_projection[:, 1], hue=clusters_average_20, palette='viridis', s=50)
plt.title("UMAP Projection with 20 Clusters (Average Linkage)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid()
plt.tight_layout()
plt.show()