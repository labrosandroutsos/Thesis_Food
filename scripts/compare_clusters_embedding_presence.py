import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# # Step 1: Presence-based clustering
# df_foodb = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_presence/foodname_compound_presence_0_1.csv', sep=';', index_col=0)

# # Perform hierarchical clustering using complete linkage
# linkage_average = linkage(df_foodb, method='complete')
# clusters_average = fcluster(linkage_average, criterion='maxclust', t=10)

# Step 2: Embedding-based clustering
# final_df = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/final_unified_embeddings_aggregated_250.pkl')

# First, we need to recluster the embeddings using 13 clusters for comparison with the reorganized presence clusters
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Load the presence-based clusters from the reorganized_clusters.txt file
# reorganized_clusters_path = 'C:/Users/labro/Downloads/revised_reorganized_clusters.txt'
# reorganized_clusters_path = 'C:/Users/labro/Downloads/final_fully_coherent_clusters.txt'
# reorganized_clusters_path = '../filtered_reorganized_clusters.txt'
reorganized_clusters_path = 'filtered_flavordb_clusters.txt'
# reorganized_clusters_path = 'processed_flavordb_clusters.txt'
presence_clusters_dict = {}

with open(reorganized_clusters_path, 'r') as file:
    current_cluster = None
    # for line in file:
    #     if "Cluster" in line:
    #         current_cluster = int(line.strip().split(" ")[1].replace(":", ""))
    #         presence_clusters_dict[current_cluster] = []
    #     elif line.strip():  # Add food name to the current cluster
    #         presence_clusters_dict[current_cluster].append(line.strip())
    import re
    def split_foods(line):
        # Use a regular expression to split on commas that are not within parentheses
        return re.split(r',\s*(?![^()]*\))', line)

    for line in file:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        if line.startswith('Cluster'):
            current_cluster = line.split('Cluster ')[-1].strip(':')
            presence_clusters_dict[current_cluster] = []
        elif line and current_cluster is not None:
            # if is_reorganized:
            #     # For reorganized clusters, each food is on a new line
            #     presence_clusters_dict[current_cluster].append(line)
            # else:
            # For FlavorDB clusters, foods are comma-separated
            foods = [food.strip() for food in split_foods(line)]
            presence_clusters_dict[current_cluster].extend(foods)

# Convert presence-based cluster dict to a DataFrame format (food_name and presence_cluster columns)
presence_clusters_list = [(food, cluster) for cluster, foods in presence_clusters_dict.items() for food in foods]
presence_clusters_df = pd.DataFrame(presence_clusters_list, columns=["food_name", "presence_cluster"])


def analyze_cluster_correspondence(presence_labels, predicted_labels, food_names):
    """
    Analyze how clusters correspond between two clustering methods
    """
    # Create DataFrame with both labelings
    df = pd.DataFrame({
        'food': food_names,
        'presence': presence_labels,
        'predicted': predicted_labels
    })
    
    # Create correspondence matrix
    correspondence = pd.crosstab(df['presence'], df['predicted'])
    
    # Calculate percentage of overlap
    correspondence_pct = correspondence.div(correspondence.sum(axis=1), axis=0) * 100
    
    print("\nCluster Correspondence (%):")
    print(correspondence_pct)
    
    # Find best matching clusters
    best_matches = []
    for presence_cluster in correspondence_pct.index:
        best_match = correspondence_pct.loc[presence_cluster].idxmax()
        overlap = correspondence_pct.loc[presence_cluster, best_match]
        best_matches.append({
            'presence_cluster': presence_cluster,
            'predicted_cluster': best_match,
            'overlap_percentage': overlap
        })
    
    print("\nBest Matching Clusters:")
    for match in best_matches:
        print(f"Presence Cluster {match['presence_cluster']} → "
              f"Predicted Cluster {match['predicted_cluster']} "
              f"(Overlap: {match['overlap_percentage']:.1f}%)")
    
    return correspondence_pct

from scipy.optimize import linear_sum_assignment
import numpy as np

def clustering_precision_recall_hungarian(true_labels, pred_labels):
    """
    Calculate precision, recall, and F1 score for clustering results using the Hungarian algorithm,
    handling different cluster orderings and labels.
    """
    # # Original LabelEncoder approach (commented out)
    # le_true, le_pred = LabelEncoder(), LabelEncoder()
    # true_labels_encoded = le_true.fit_transform(true_labels)
    # pred_labels_encoded = le_pred.fit_transform(pred_labels)
    # cm = confusion_matrix(true_labels_encoded, pred_labels_encoded)

    # New approach with explicit index mapping
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    
    # Create mapping dictionaries
    true_to_idx = {label: idx for idx, label in enumerate(unique_true)}
    pred_to_idx = {label: idx for idx, label in enumerate(unique_pred)}
    
    # Convert labels to indices
    true_indices = np.array([true_to_idx[label] for label in true_labels])
    pred_indices = np.array([pred_to_idx[label] for label in pred_labels])
    
    # Compute confusion matrix using indices
    n_true_clusters = len(unique_true)
    n_pred_clusters = len(unique_pred)
    cm = confusion_matrix(true_indices, pred_indices,
                         labels=range(max(n_true_clusters, n_pred_clusters)))

    # Pad the confusion matrix if necessary
    if cm.shape[0] != cm.shape[1]:
        max_dim = max(cm.shape[0], cm.shape[1])
        padded_cm = np.zeros((max_dim, max_dim))
        padded_cm[:cm.shape[0], :cm.shape[1]] = cm
        cm = padded_cm
    
    # Create correspondence matrix for visualization
    df = pd.DataFrame({
        'true': true_labels,
        'pred': pred_labels
    })
    
    # Create and print raw counts correspondence
    correspondence_raw = pd.crosstab(df['true'], df['pred'])
    print("\nCluster Correspondence (Raw Counts):")
    print(correspondence_raw)
    
    # Calculate and print percentage overlap from true cluster perspective
    correspondence_pct_true = correspondence_raw.div(correspondence_raw.sum(axis=1), axis=0) * 100
    print("\nCluster Correspondence (% of True Cluster):")
    print(correspondence_pct_true)
    
    # Calculate and print percentage overlap from predicted cluster perspective
    correspondence_pct_pred = correspondence_raw.div(correspondence_raw.sum(axis=0), axis=1) * 100
    print("\nCluster Correspondence (% of Predicted Cluster):")
    print(correspondence_pct_pred)

    # Hungarian algorithm to maximize cluster alignment
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    
    # Print the optimal matching
    print("\nOptimal Cluster Matching:")
    for i, j in zip(row_ind, col_ind):
        if i < len(unique_true) and j < len(unique_pred):  # Only show valid matches
            true_cluster = unique_true[i]
            pred_cluster = unique_pred[j]
            overlap = cm[i, j]
            total_true = cm[i, :].sum()
            total_pred = cm[:, j].sum()
            if total_true > 0 and total_pred > 0:  # Only show non-empty clusters
                print(f"True Cluster '{true_cluster}' matched with Predicted Cluster '{pred_cluster}'")
                print(f"  Overlap: {overlap} items")
                print(f"  Coverage: {overlap/total_true*100:.1f}% of true cluster, {overlap/total_pred*100:.1f}% of predicted cluster")

    # Calculate precision and recall for each matched cluster
    precision_per_cluster = []
    recall_per_cluster = []
    
    for i, j in zip(row_ind, col_ind):
        if i < len(unique_true) and j < len(unique_pred):  # Only consider valid matches
            total_pred = cm[:, j].sum()
            total_true = cm[i, :].sum()
            if total_pred > 0 and total_true > 0:  # Only consider non-empty clusters
                precision = cm[i, j] / total_pred
                recall = cm[i, j] / total_true
                precision_per_cluster.append(precision)
                recall_per_cluster.append(recall)

    # Macro-averaged precision and recall
    macro_precision = np.mean(precision_per_cluster) if precision_per_cluster else 0
    macro_recall = np.mean(recall_per_cluster) if recall_per_cluster else 0

    # Calculate F1 score
    if macro_precision + macro_recall > 0:
        f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    else:
        f1_score = 0.0
    
    print(f"\nOverall Metrics:")
    print(f"Macro Precision: {macro_precision:.3f}")
    print(f"Macro Recall: {macro_recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")

    return macro_precision, macro_recall, f1_score

def clustering_precision_recall(true_labels, pred_labels):
    """
    Calculate precision, recall, and F1 score for clustering results using cluster correspondence
    
    Parameters:
    -----------
    true_labels : array-like
        Ground truth cluster labels
    pred_labels : array-like
        Predicted cluster labels
        
    Returns:
    --------
    tuple
        (macro_precision, macro_recall, f1_score)
    """
    # Create correspondence matrix
    df = pd.DataFrame({
        'true': true_labels,
        'pred': pred_labels
    })
    correspondence = pd.crosstab(df['true'], df['pred'])
    
    # Calculate percentage of overlap
    correspondence_pct = correspondence.div(correspondence.sum(axis=1), axis=0) * 100
    
    print("\nCluster Correspondence (%):")
    print(correspondence_pct)
    
    # Find best matching clusters
    precision_per_cluster = []
    recall_per_cluster = []
    
    for true_cluster in correspondence.index:
        # Find best matching predicted cluster
        best_pred_cluster = correspondence.loc[true_cluster].idxmax()
        
        # Calculate precision: how many of predicted cluster members actually belong to true cluster
        precision = correspondence.loc[true_cluster, best_pred_cluster] / correspondence[best_pred_cluster].sum()
        
        # Calculate recall: how many of true cluster members were captured in predicted cluster
        recall = correspondence.loc[true_cluster, best_pred_cluster] / correspondence.loc[true_cluster].sum()
        
        precision_per_cluster.append(precision)
        recall_per_cluster.append(recall)
    
    # Calculate macro averages
    macro_precision = np.mean(precision_per_cluster)
    macro_recall = np.mean(recall_per_cluster)
    
    # Calculate F1 score
    if macro_precision + macro_recall > 0:
        f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    else:
        f1_score = 0.0
        
    return macro_precision, macro_recall, f1_score

def evaluate_clustering_algorithms(X_flat, presence_clusters_df, final_df, embedding_size):
    """
    Evaluate different clustering algorithms with multiple metrics
    """
    results = []
    correspondences = {}  # Initialize correspondences at function level
    
    # Initialize lists to store results for correspondence analysis
    kmeans_results = []
    hierarchical_results = []
    dbscan_results = []
    
    # Scale the data for DBSCAN and metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    def calculate_metrics(labels, X_data, algorithm_name, n_clusters, noise_points=0):
        """Helper function to calculate all metrics"""
        total_points = len(X_data)
        
        # For DBSCAN, calculate internal metrics only on non-noise points
        if 'DBSCAN' in algorithm_name and noise_points > 0:
            # Filter out noise points for internal metrics
            valid_mask = labels != -1
            valid_X = X_data[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(set(valid_labels)) >= 2:  # Need at least 2 clusters for internal metrics
                try:
                    silhouette = silhouette_score(valid_X, valid_labels)
                    calinski = calinski_harabasz_score(valid_X, valid_labels)
                    davies = davies_bouldin_score(valid_X, valid_labels)
                except:
                    silhouette = None
                    calinski = None
                    davies = None
            else:
                silhouette = None
                calinski = None
                davies = None
        else:
            try:
                silhouette = silhouette_score(X_data, labels)
                calinski = calinski_harabasz_score(X_data, labels)
                davies = davies_bouldin_score(X_data, labels)
            except:
                silhouette = None
                calinski = None
                davies = None
        
        # Create comparison DataFrame for supervised metrics
        temp_df = final_df[['food_name']].copy()
        temp_df['cluster'] = labels
        comparison_df = pd.merge(presence_clusters_df, temp_df, on='food_name')
        
        # Calculate supervised metrics
        ari = adjusted_rand_score(comparison_df['presence_cluster'], comparison_df['cluster'])
        nmi = normalized_mutual_info_score(comparison_df['presence_cluster'], comparison_df['cluster'])
        homogeneity = homogeneity_score(comparison_df['presence_cluster'], comparison_df['cluster'])
        completeness = completeness_score(comparison_df['presence_cluster'], comparison_df['cluster'])
        v_measure = v_measure_score(comparison_df['presence_cluster'], comparison_df['cluster'])
        
        # Calculate precision and recall
        precision, recall, f1 = clustering_precision_recall_hungarian(
            comparison_df['presence_cluster'], 
            comparison_df['cluster']
        )
    
        
        # Store results for correspondence analysis
        if 'DBSCAN' in algorithm_name:
            dbscan_results.append((algorithm_name, ari, labels))
        elif 'K-means' in algorithm_name:
            kmeans_results.append((algorithm_name, ari, labels))
        elif 'Hierarchical' in algorithm_name:
            hierarchical_results.append((algorithm_name, ari, labels))
        
        result = {
            'embedding_size': embedding_size,
            'algorithm': algorithm_name,
            'n_clusters': n_clusters,
            'noise_points': noise_points if noise_points else 0,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'ari': ari,
            'nmi': nmi,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_points': total_points,
            'noise_percentage': (noise_points / total_points) * 100 if total_points > 0 else 0,
            'clustered_points': total_points - noise_points,
            'clustered_percentage': ((total_points - noise_points) / total_points) * 100 if total_points > 0 else 0
        }
        return result
    
    def analyze_top_models(model_results, model_type, top_k=3):
        """Analyze top performing models based on ARI score"""
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'food_name': final_df['food_name'],
            'presence_cluster': presence_clusters_df['presence_cluster'],
            'cluster': None  # Will be filled with predicted labels
        })
        
        if model_results:
            # Sort results by ARI score
            model_results.sort(key=lambda x: x[1], reverse=True)
            # Take top k models
            top_k = min(top_k, len(model_results))
            for i in range(top_k):
                algorithm_name, _, labels = model_results[i]
                print(f"\nAnalyzing cluster correspondence for {algorithm_name} (embedding_size={embedding_size}) - Top {i+1} {model_type}:")
                
                # Update cluster labels in comparison_df
                comparison_df['cluster'] = labels
                
                if model_type == 'DBSCAN' and len(set(labels)) - (1 if -1 in labels else 0) >= 2:
                    # Filter out noise points for DBSCAN
                    valid_mask = labels != -1
                    filtered_presence = comparison_df['presence_cluster'][valid_mask]
                    filtered_clusters = comparison_df['cluster'][valid_mask]
                    filtered_foods = comparison_df['food_name'][valid_mask]
                    correspondence = analyze_cluster_correspondence(
                        filtered_presence,
                        filtered_clusters,
                        filtered_foods
                    )
                else:
                    correspondence = analyze_cluster_correspondence(
                        comparison_df['presence_cluster'],
                        comparison_df['cluster'],
                        comparison_df['food_name']
                    )
                if correspondence is not None:
                    correspondences[f"{algorithm_name}_{embedding_size}"] = correspondence
                    print(f"Added correspondence for {algorithm_name}_{embedding_size}")
    
    # First collect all results
    # 1. K-means
    cluster_numbers = [10, 15, 20, 25, 30, 35]
    for n_clusters in cluster_numbers:
        kmeans = KMeans(n_clusters=n_clusters, n_init=50, init='k-means++', random_state=42)
        kmeans_labels = kmeans.fit_predict(X_flat)
        result = calculate_metrics(kmeans_labels, X_flat, f'K-means++ (n={n_clusters})', n_clusters)
        if result is not None:
            results.append(result)
            kmeans_results.append((f'K-means++ (n={n_clusters})', result['ari'], kmeans_labels))
    
    # 2. Hierarchical Clustering
    for n_clusters in cluster_numbers:
        for linkage in ['ward', 'complete', 'average', 'single']:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            hierarchical_labels = hierarchical.fit_predict(X_flat)
            result = calculate_metrics(hierarchical_labels, X_flat, f'Hierarchical ({linkage}, n={n_clusters})', n_clusters)
            if result is not None:
                results.append(result)
                hierarchical_results.append((f'Hierarchical ({linkage}, n={n_clusters})', result['ari'], hierarchical_labels))
    
    # 3. DBSCAN
    eps_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    min_samples_values = [2, 3, 4, 5]
    metrics = ['euclidean', 'cosine']
    dbscan_params = []
    
    for metric in metrics:
        for eps in eps_values:
            for min_samples in min_samples_values:
                X_use = X_scaled if metric == 'euclidean' else X_flat
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                dbscan_labels = dbscan.fit_predict(X_use)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                noise_points = sum(1 for x in dbscan_labels if x == -1)
                
                if n_clusters >= 2:  # Only require at least 2 clusters for metrics to be meaningful
                    result = calculate_metrics(
                        dbscan_labels,
                        X_use,
                        f'DBSCAN ({metric}, eps={eps}, min_samples={min_samples})',
                        n_clusters,
                        noise_points
                    )
                    if result is not None:
                        results.append(result)
                        dbscan_results.append((f'DBSCAN ({metric}, eps={eps}, min_samples={min_samples})', result['ari'], dbscan_labels))
                        dbscan_params.append((metric, eps, min_samples))
    
    # Now analyze top models after all results are collected
    # print("\nAnalyzing top models for each clustering method...")
    # if kmeans_results:
    #     print(f"\nAnalyzing {len(kmeans_results)} K-means results")
    #     analyze_top_models(kmeans_results, 'K-means', top_k=2)
    # if hierarchical_results:
    #     print(f"\nAnalyzing {len(hierarchical_results)} Hierarchical results")
    #     analyze_top_models(hierarchical_results, 'Hierarchical', top_k=3)
    # if dbscan_results:
    #     print(f"\nAnalyzing {len(dbscan_results)} DBSCAN results")
    #     analyze_top_models(dbscan_results, 'DBSCAN', top_k=3)
    
    # print(f"\nNumber of correspondences collected: {len(correspondences)}")
    # for key in correspondences.keys():
    #     print(f"Correspondence found for: {key}")
    
    # return pd.DataFrame(results), correspondences
    return pd.DataFrame(results)

# List of embedding sizes to test
embedding_sizes = [50, 100, 150, 200, 250]
all_results = []
all_correspondences = {}

# Test each embedding size
for size in embedding_sizes:
    print(f"\nProcessing embedding size: {size}")
    # final_df = pd.read_pickle(f'C:/Users/labro/Downloads/Thesis_Food/embeddings/final_unified_embeddings_aggregated_{size}.pkl')
    final_df = pd.read_pickle(f'C:/Users/labro/Downloads/Thesis_Food/embeddings/filtered_unified_embeddings_aggregated_{size}.pkl')
    X_flat = np.array(final_df['unified_embedding'].tolist())
    
    # results_df, correspondences = evaluate_clustering_algorithms(X_flat, presence_clusters_df, final_df, size)
    results_df = evaluate_clustering_algorithms(X_flat, presence_clusters_df, final_df, size)
    all_results.append(results_df)
    # all_correspondences.update(correspondences)

# Combine all results
final_results = pd.concat(all_results, ignore_index=True)

# Print results sorted by different metrics
print("\nResults sorted by different metrics:")
metrics = ['ari', 'nmi', 'v_measure', 'homogeneity', 'completeness', 'precision', 'recall', 'f1']
for metric in metrics:
    print(f"\nSorted by {metric}:")
    pivot = pd.pivot_table(
        final_results,
        values=metric,
        index='embedding_size',
        columns='algorithm',
        aggfunc='first'
    )
    print(pivot)

# Save detailed results
final_results.to_csv("clustering_algorithm_comparison_flavordb_VS_all_embeddings_precision_recall_withcorrespondence_original_foods_PROCESSEDFOODDB.csv", index=False)

# print('correspondences')
# print(all_correspondences)
# # Save correspondences
# for name, corr in all_correspondences.items():
#     if corr is not None:  # Only save if correspondence exists
#         corr.to_csv(f"cluster_correspondence_{name}_flavordb_VS_embeddings_original_foods.csv")
