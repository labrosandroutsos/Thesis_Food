from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import numpy as np

df_foodb = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_presence/foodname_compound_presence_0_1_filtered.csv', sep=';', index_col=0)

# Compute linkage matrices for each type
linkage_single = linkage(df_foodb, method='single')
linkage_complete = linkage(df_foodb, method='complete')
linkage_average = linkage(df_foodb, method='average')

# Test different numbers of clusters
n_clusters_list = [10, 20, 30, 40]
results = []

for n_clusters in n_clusters_list:
    # Get cluster labels for each linkage type
    clusters_single = fcluster(linkage_single, criterion='maxclust', t=n_clusters)
    clusters_complete = fcluster(linkage_complete, criterion='maxclust', t=n_clusters)
    clusters_average = fcluster(linkage_average, criterion='maxclust', t=n_clusters)
    
    # Calculate metrics for single linkage
    silhouette_single = silhouette_score(df_foodb, clusters_single)
    davies_bouldin_single = davies_bouldin_score(df_foodb, clusters_single)
    calinski_harabasz_single = calinski_harabasz_score(df_foodb, clusters_single)
    
    # Calculate metrics for complete linkage
    silhouette_complete = silhouette_score(df_foodb, clusters_complete)
    davies_bouldin_complete = davies_bouldin_score(df_foodb, clusters_complete)
    calinski_harabasz_complete = calinski_harabasz_score(df_foodb, clusters_complete)
    
    # Calculate metrics for average linkage
    silhouette_average = silhouette_score(df_foodb, clusters_average)
    davies_bouldin_average = davies_bouldin_score(df_foodb, clusters_average)
    calinski_harabasz_average = calinski_harabasz_score(df_foodb, clusters_average)
    
    # Store results
    results.extend([
        {'n_clusters': n_clusters, 'linkage': 'Single', 
         'silhouette': silhouette_single, 
         'davies_bouldin': davies_bouldin_single,
         'calinski_harabasz': calinski_harabasz_single},
        {'n_clusters': n_clusters, 'linkage': 'Complete', 
         'silhouette': silhouette_complete, 
         'davies_bouldin': davies_bouldin_complete,
         'calinski_harabasz': calinski_harabasz_complete},
        {'n_clusters': n_clusters, 'linkage': 'Average', 
         'silhouette': silhouette_average, 
         'davies_bouldin': davies_bouldin_average,
         'calinski_harabasz': calinski_harabasz_average}
    ])

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Print results grouped by metric
print("\nSilhouette Scores (higher is better):")
print(results_df.pivot(index='n_clusters', columns='linkage', values='silhouette').round(4))

print("\nDavies-Bouldin Index (lower is better):")
print(results_df.pivot(index='n_clusters', columns='linkage', values='davies_bouldin').round(4))

print("\nCalinski-Harabasz Index (higher is better):")
print(results_df.pivot(index='n_clusters', columns='linkage', values='calinski_harabasz').round(4))

# Find the best configuration for each metric
best_silhouette = results_df.loc[results_df['silhouette'].idxmax()]
best_davies_bouldin = results_df.loc[results_df['davies_bouldin'].idxmin()]
best_calinski = results_df.loc[results_df['calinski_harabasz'].idxmax()]

print("\nBest configurations:")
print(f"Best Silhouette Score: {best_silhouette['linkage']} linkage with {best_silhouette['n_clusters']} clusters (score: {best_silhouette['silhouette']:.4f})")
print(f"Best Davies-Bouldin: {best_davies_bouldin['linkage']} linkage with {best_davies_bouldin['n_clusters']} clusters (score: {best_davies_bouldin['davies_bouldin']:.4f})")
print(f"Best Calinski-Harabasz: {best_calinski['linkage']} linkage with {best_calinski['n_clusters']} clusters (score: {best_calinski['calinski_harabasz']:.4f})")

# Save the best clustering result (Average linkage with 10 clusters)
best_clusters = fcluster(linkage_average, criterion='maxclust', t=10)

# Create a DataFrame with food names and their cluster assignments
cluster_assignments = pd.DataFrame({
    'Food': df_foodb.index,
    'Cluster': best_clusters
})

# Sort by cluster number and food name for better readability
cluster_assignments = cluster_assignments.sort_values(['Cluster', 'Food'])

# Save to text file in the requested format
output_path = 'C:/Users/labro/Downloads/Thesis_Food/compounds_presence/average_linkage_clusters.txt'
with open(output_path, 'w') as f:
    for cluster_id in range(1, max(best_clusters) + 1):
        # Get foods in current cluster
        cluster_foods = cluster_assignments[cluster_assignments['Cluster'] == cluster_id]['Food']
        
        # Write cluster header
        f.write(f"Cluster {cluster_id}:\n")
        
        # Write foods in alphabetical order
        for food in sorted(cluster_foods):
            f.write(f"{food}\n")
        
        # Add blank line between clusters
        f.write("\n")

print("\nCluster assignments have been saved to:", output_path)

# Print summary of cluster sizes
cluster_sizes = cluster_assignments['Cluster'].value_counts().sort_index()
print("\nCluster sizes:")
for cluster_id, size in cluster_sizes.items():
    print(f"Cluster {cluster_id}: {size} foods")