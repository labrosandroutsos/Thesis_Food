import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, SpectralClustering # Added SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, mutual_info_score # Added mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import re
from joblib import Parallel, delayed # Import joblib
from sklearn.metrics import fowlkes_mallows_score
from bcubed_metrics.bcubed import Bcubed
from sklearn.preprocessing import normalize  

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
from bcubed_metrics.bcubed import Bcubed

# Load the presence-based clusters from the reorganized_clusters.txt file
# reorganized_clusters_path = 'C:/Users/labro/Downloads/revised_reorganized_clusters.txt'
# reorganized_clusters_path = 'C:/Users/labro/Downloads/final_fully_coherent_clusters.txt'
# reorganized_clusters_path = '/home/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/filtered_reorganized_clusters.txt'
# reorganized_clusters_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/flavordb_clusters.txt'
# reorganized_clusters_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/filtered_flavordb_clusters.txt'
reorganized_clusters_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/filtered_flavordb_clusters_683.txt'
# reorganized_clusters_path = '/home/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/processed_flavordb_clusters.txt'
presence_clusters_dict = {}

def fmi_pair_counting(true_labels, pred_labels):
    """
    Calculate Fowlkes-Mallows Index (FMI) using pair counting
    """
    return fowlkes_mallows_score(true_labels, pred_labels)
    
# def bcubed_scores(true_labels, pred_labels, averaging="micro"):
#     """
#     B-cubed precision, recall, F1.
#     averaging ∈ {"micro", "macro"}  (micro is the usual choice).
#     """
#     # Convert labels to dictionary format expected by Bcubed
#     # {item_id: {str(cluster_label)}} - labels are converted to strings and wrapped in a set.
#     true_clusters_dict = {i: {str(label)} for i, label in enumerate(true_labels)}
#     pred_clusters_dict = {i: {str(label)} for i, label in enumerate(pred_labels)}

#     print(true_clusters_dict)
#     print(pred_clusters_dict)
#     bcubed = Bcubed(predicted_clustering=pred_clusters_dict, ground_truth_clustering=true_clusters_dict)
#     metrics = bcubed.get_metrics()
#     print(metrics)
#     prec = metrics['precision']
#     print(f"B-cubed Precision: {prec}")
#     rec  = metrics['recall']
#     f1   = metrics['fscore']
#     return prec, rec, f1

def split_foods(line):
    # Use a regular expression to split on commas that are not within parentheses
    return re.split(r',\s*(?![^()]*\))', line)

with open(reorganized_clusters_path, 'r') as file:
    current_cluster = None
    for line in file:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.startswith('Cluster'):
            current_cluster = line.split('Cluster ')[-1].strip(':')
            presence_clusters_dict[current_cluster] = []
        elif line and current_cluster is not None:
            foods = [food.strip() for food in split_foods(line)]
            presence_clusters_dict[current_cluster].extend(foods)

# Convert presence-based cluster dict to a DataFrame format
presence_clusters_list = [(food, cluster) for cluster, foods in presence_clusters_dict.items() for food in foods]
presence_clusters_df = pd.DataFrame(presence_clusters_list, columns=["food_name", "presence_cluster"])
print(f"Loaded {presence_clusters_df.shape[0]} foods from presence cluster file {reorganized_clusters_path}")
print(f"Unique foods in presence clusters: {presence_clusters_df.food_name.nunique()}")

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
        # Handle cases where a presence cluster might not have any predicted matches
        if presence_cluster in correspondence_pct.index and not correspondence_pct.loc[presence_cluster].empty:
             best_match = correspondence_pct.loc[presence_cluster].idxmax()
             overlap = correspondence_pct.loc[presence_cluster, best_match]
             best_matches.append({
                 'presence_cluster': presence_cluster,
                 'predicted_cluster': best_match,
                 'overlap_percentage': overlap
             })
        else:
             print(f"Warning: Presence cluster '{presence_cluster}' has no corresponding predicted clusters in this subset.")

    
    # print("\nBest Matching Clusters:")
    # for match in best_matches:
    #     print(f"Presence Cluster {match['presence_cluster']} → "
    #           f"Predicted Cluster {match['predicted_cluster']} "
    #           f"(Overlap: {match['overlap_percentage']:.1f}%)")
    
    return correspondence_pct

def clustering_precision_recall_hungarian(true_labels, pred_labels):
    """
    Calculate precision, recall, and F1 score for clustering results using the Hungarian algorithm,
    handling different cluster orderings and labels.
    """
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
    # Ensure labels cover all possible indices for both true and pred
    all_indices = list(range(max(n_true_clusters, n_pred_clusters)))
    cm = confusion_matrix(true_indices, pred_indices, labels=all_indices)

    # Pad the confusion matrix if necessary (already handled by labels=all_indices)
    # if cm.shape[0] != cm.shape[1]:
    #     max_dim = max(cm.shape[0], cm.shape[1])
    #     padded_cm = np.zeros((max_dim, max_dim))
    #     padded_cm[:cm.shape[0], :cm.shape[1]] = cm
    #     cm = padded_cm
    
    # Create correspondence matrix for visualization
    df = pd.DataFrame({
        'true': true_labels,
        'pred': pred_labels
    })
    
    # Create and print raw counts correspondence
    correspondence_raw = pd.crosstab(df['true'], df['pred'], dropna=False) # Keep all clusters
    # print("\nCluster Correspondence (Raw Counts):")
    # print(correspondence_raw)
    
    # Calculate and print percentage overlap from true cluster perspective
    correspondence_pct_true = correspondence_raw.div(correspondence_raw.sum(axis=1), axis=0).fillna(0) * 100
    # print("\nCluster Correspondence (% of True Cluster):")
    # print(correspondence_pct_true)
    
    # Calculate and print percentage overlap from predicted cluster perspective
    correspondence_pct_pred = correspondence_raw.div(correspondence_raw.sum(axis=0), axis=1).fillna(0) * 100
    # print("\nCluster Correspondence (% of Predicted Cluster):")
    # print(correspondence_pct_pred)

    # Hungarian algorithm to maximize cluster alignment
    # Use the cost matrix (negative counts) for maximization with linear_sum_assignment
    cost_matrix = -cm 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # ──────────────────────────────────────────────────────────────────────────
    # 🔗  MICRO-averaged precision / recall / F1  (pair counts, not macro)
    # -------------------------------------------------------------------------
    #  TP = items that fall into a matched (true,pred) pair
    TP = cm[row_ind, col_ind].sum()

    #  FP = all items assigned to those predicted clusters minus TP
    FP = cm[:, col_ind].sum() - TP

    #  FN = all items in those true clusters minus TP
    FN = cm[row_ind, :].sum() - TP

    micro_precision = TP / (TP + FP) if TP + FP else 0.0
    micro_recall    = TP / (TP + FN) if TP + FN else 0.0
    if micro_precision + micro_recall:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    # print(f"\nMicro Precision: {micro_precision:.3f}")
    # print(f"Micro Recall:    {micro_recall:.3f}")
    # print(f"Micro F1:        {micro_f1:.3f}")

    # Print the optimal matching
    # print("\nOptimal Cluster Matching:") # Commented out to reduce console output for 10 runs
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        # Check if the indices are within the bounds of actual unique labels
        if i < n_true_clusters and j < n_pred_clusters:
            true_cluster = unique_true[i]
            pred_cluster = unique_pred[j]
            overlap = cm[i, j]
            total_true = cm[i, :].sum()
            total_pred = cm[:, j].sum()
            if total_true > 0 and total_pred > 0: 
                 matched_pairs.append((i, j)) # Store valid matched indices
                #  print(f"True Cluster '{true_cluster}' matched with Predicted Cluster '{pred_cluster}'")
                #  print(f"  Overlap: {overlap} items")
                #  print(f"  Coverage: {overlap/total_true*100:.1f}% of true cluster, {overlap/total_pred*100:.1f}% of predicted cluster")
            # else:
                 # Optional: print info about empty clusters being ignored in matching
                 # print(f"  (Ignoring match involving empty true/pred cluster: True='{true_cluster}', Pred='{pred_cluster}')")
        # else:
            # Optional: print info about matches involving padding indices
            # print(f"  (Ignoring match involving padded index: True={i}, Pred={j})")


    # Calculate precision and recall based ONLY on the optimally matched, non-empty clusters
    precision_per_cluster = []
    recall_per_cluster = []
    
    for i, j in matched_pairs: # Iterate through valid matched pairs
        total_pred = cm[:, j].sum()
        total_true = cm[i, :].sum()
        # We already checked total_true > 0 and total_pred > 0 to add to matched_pairs
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
    
    # print(f"\nOverall Metrics (based on {len(matched_pairs)} matched pairs):")
    # print(f"Macro Precision: {macro_precision:.3f}")
    # print(f"Macro Recall: {macro_recall:.3f}")
    # print(f"F1 Score: {f1_score:.3f}")

    return macro_precision, macro_recall, f1_score, micro_precision, micro_recall, micro_f1

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
    
    # print("\nCluster Correspondence (%):")
    # print(correspondence_pct)
    
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

def variation_of_information(true, pred):
    h_true = mutual_info_score(true, true)
    h_pred = mutual_info_score(pred, pred)
    mi     = mutual_info_score(true, pred)
    return h_true + h_pred - 2*mi

def evaluate_clustering_algorithms(X_flat, presence_clusters_df_global, final_df_local, embedding_size, window, run_seed):
    """
    Evaluate different clustering algorithms with multiple metrics
    """
    results = []
    # correspondences = {} # Commented out for 10-run averaging
    
    # Commented out for 10-run averaging
    # kmeans_results = []
    # hierarchical_results = []
    # dbscan_results = []
    # spectral_results = [] 
    
    # Scale the data for DBSCAN and metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    def calculate_metrics(labels, X_data, algorithm_name, n_clusters, noise_points=0):
        """Helper function to calculate all metrics"""
        total_points = len(X_data)
        silhouette, calinski, davies = None, None, None # Initialize internal metrics
        
        # Calculate internal metrics only on non-noise points for DBSCAN, or all points otherwise
        try:
            if 'DBSCAN' in algorithm_name and noise_points > 0:
                valid_mask = labels != -1
                if np.sum(valid_mask) > 1: # Need at least 2 non-noise points
                     valid_X = X_data[valid_mask]
                     valid_labels = labels[valid_mask]
                     if len(set(valid_labels)) >= 2:  # Need at least 2 clusters among non-noise points
                         silhouette = silhouette_score(valid_X, valid_labels)
                         calinski = calinski_harabasz_score(valid_X, valid_labels)
                         davies = davies_bouldin_score(valid_X, valid_labels)
            elif len(set(labels)) - (1 if -1 in labels else 0) >= 2: # Need at least 2 clusters (excluding potential noise label -1)
                 # Calculate for all points if not DBSCAN or DBSCAN with no noise
                 silhouette = silhouette_score(X_data, labels)
                 calinski = calinski_harabasz_score(X_data, labels)
                 davies = davies_bouldin_score(X_data, labels)
        except ValueError as e:
             print(f"Could not calculate internal metrics for {algorithm_name}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during internal metric calculation for {algorithm_name}: {e}")
        
        # Create comparison DataFrame for supervised metrics
        temp_df = final_df_local[['food_name']].copy()
        temp_df['cluster'] = labels
        comparison_df_merged = pd.merge(presence_clusters_df_global, temp_df, on='food_name', how='inner')
        
        if comparison_df_merged.empty or comparison_df_merged['presence_cluster'].nunique() < 1 or comparison_df_merged['cluster'].nunique() < 1:
            print(f"Warning: Insufficient data for supervised metrics comparison for {algorithm_name}. Skipping.")
            ari, nmi, homogeneity, completeness, v_measure = None, None, None, None, None
            precision, recall, f1 = None, None, None
            micro_precision, micro_recall, micro_f1, fmi, vi = None, None, None, None, None # Added vi
        else:
            true_labels_common = comparison_df_merged['presence_cluster']
            pred_labels_common = comparison_df_merged['cluster']
            vi = None # Initialize vi
            try:
                ari = adjusted_rand_score(true_labels_common, pred_labels_common)
                nmi = normalized_mutual_info_score(true_labels_common, pred_labels_common)
                homogeneity = homogeneity_score(true_labels_common, pred_labels_common)
                completeness = completeness_score(true_labels_common, pred_labels_common)
                v_measure = v_measure_score(true_labels_common, pred_labels_common)
                precision, recall, f1, micro_precision, micro_recall, micro_f1 = clustering_precision_recall_hungarian(
                    true_labels_common, 
                    pred_labels_common
                )
                # NEW ① Pair–counting FMI
                fmi = fmi_pair_counting(true_labels_common, pred_labels_common)
                # Calculate Variation of Information
                vi = variation_of_information(true_labels_common, pred_labels_common)
                # bcubed_prec, bcubed_rec, bcubed_f1 = bcubed_scores(pred_labels_common, true_labels_common, "micro")

            except ValueError as e:
                print(f"Could not calculate supervised metrics for {algorithm_name}: {e}")
                ari, nmi, homogeneity, completeness, v_measure = None, None, None, None, None
                precision, recall, f1, micro_precision, micro_recall, micro_f1, fmi, vi = None, None, None, None, None, None, None, None # Added vi
            except Exception as e:
                 print(f"An unexpected error occurred during supervised metric calculation for {algorithm_name}: {e}")
                 ari, nmi, homogeneity, completeness, v_measure = None, None, None, None, None
                 precision, recall, f1, micro_precision, micro_recall, micro_f1, fmi, vi = None, None, None, None, None, None, None, None # Added vi
    
        # Store results for correspondence analysis if supervised metrics were calculated - Commented out for 10-run averaging
        # if ari is not None: 
        #     result_tuple = (algorithm_name, ari, precision, labels, comparison_df_merged['food_name'], true_labels_common, pred_labels_common)
            
        #     if 'DBSCAN' in algorithm_name:
        #         dbscan_results.append(result_tuple)
        #     elif 'K-means' in algorithm_name:
        #         kmeans_results.append(result_tuple)
        #     elif 'Hierarchical' in algorithm_name:
        #         hierarchical_results.append(result_tuple)
        #     elif 'Spectral' in algorithm_name: 
        #         spectral_results.append(result_tuple)


        result = {
            'embedding_size': embedding_size,
            'window_size': window,
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
            'fmi': fmi,
            'vi': vi, # Added vi
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'total_points_in_embedding': total_points, 
            'compared_points': len(comparison_df_merged), 
            'noise_percentage': (noise_points / total_points) * 100 if total_points > 0 else 0,
            'clustered_points': total_points - noise_points,
            'clustered_percentage': ((total_points - noise_points) / total_points) * 100 if total_points > 0 else 0
        }
        return result
    
    # Re-enabled correspondence analysis function - Commented out for 10-run averaging
    # def analyze_top_models(model_results, model_type, top_k=3):
    #     """Analyze top performing models based on Precision score and store correspondence"""
    #     if model_results:
    #         model_results.sort(key=lambda x: x[2] if x[2] is not None else -1, reverse=True) 
    #         top_k = min(top_k, len(model_results))
    #         for i in range(top_k):
    #             algorithm_name, _, precision_score, _, compared_food_names, compared_true_labels, compared_pred_labels = model_results[i] 
    #             if precision_score is None: 
    #                  print(f"Skipping correspondence analysis for {algorithm_name} (Top {i+1} {model_type}) due to missing Precision score.") 
    #                  continue 
                     
    #             print(f"\nAnalyzing cluster correspondence for {algorithm_name} (embedding_size={embedding_size}, window={window}) - Top {i+1} {model_type} (Precision: {precision_score:.3f}):") 
                
    #             # correspondence = analyze_cluster_correspondence(
    #             #     compared_true_labels,
    #             #     compared_pred_labels,
    #             #     compared_food_names
    #             # )
    #             # if correspondence is not None:
    #             #     correspondence_key = f"{algorithm_name}_size{embedding_size}_window{window}"
    #             #     correspondences[correspondence_key] = correspondence
    #             #     print(f"Added correspondence for {correspondence_key}")
    
    # --- Clustering Algorithms (Restored Parameters) --- 
    cluster_numbers = [10, 15, 20, 25, 30, 35] # Modified to evaluate multiple cluster numbers
    
    # 1. K-means
    for n_clusters in cluster_numbers:
        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=50, init='k-means++', random_state=run_seed) # Used run_seed
            kmeans_labels = kmeans.fit_predict(X_scaled) # Changed from X_flat to X_scaled
            result = calculate_metrics(kmeans_labels, X_scaled, f'K-means++ (n={n_clusters})', n_clusters) # Changed from X_flat to X_scaled
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error during K-means (n={n_clusters}): {e}")

    # 2. Hierarchical Clustering
    linkage_types = ['ward', 'complete', 'average', 'single']
    hierarchical_metrics_to_test = ['euclidean', 'cosine', 'manhattan']

    for n_clusters in cluster_numbers:
        for linkage_type in linkage_types:
            if linkage_type == 'ward':
                # Ward linkage only supports Euclidean metric
                try:
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type, metric='euclidean') # metric is effectively euclidean for ward
                    hierarchical_labels = hierarchical.fit_predict(X_scaled) # Ward uses Euclidean, so X_scaled
                    result = calculate_metrics(hierarchical_labels, X_scaled, f'Hierarchical ({linkage_type}, euclidean, n={n_clusters})', n_clusters)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error during Hierarchical Clustering (linkage={linkage_type}, metric=euclidean, n={n_clusters}): {e}")
            else:
                # For other linkage types, iterate through specified metrics
                for metric_choice in hierarchical_metrics_to_test:
                    try:
                        X_hierarchical_use = X_flat if metric_choice == 'cosine' else X_scaled # Corrected: X_flat for cosine
                        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type, metric=metric_choice)
                        hierarchical_labels = hierarchical.fit_predict(X_hierarchical_use)
                        result = calculate_metrics(hierarchical_labels, X_hierarchical_use, f'Hierarchical ({linkage_type}, {metric_choice}, n={n_clusters})', n_clusters)
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error during Hierarchical Clustering (linkage={linkage_type}, metric={metric_choice}, n={n_clusters}): {e}")

    # 3. DBSCAN
    # Restored original DBSCAN parameters
    eps_values = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] 
    min_samples_values = [2, 3, 4, 5] 
    metrics = ['euclidean', 'cosine']
    
    for metric in metrics:
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    X_use = X_flat if metric == 'cosine' else X_scaled # Corrected: X_flat for cosine
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                    dbscan_labels = dbscan.fit_predict(X_use)
                    
                    n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                    noise_points = sum(1 for x in dbscan_labels if x == -1)
                    
                    # Allow calculation if at least 1 cluster is found (can compare noise vs cluster)
                    if n_clusters_found >= 1: 
                        result = calculate_metrics(
                            dbscan_labels,
                            X_use,
                            f'DBSCAN ({metric}, eps={eps}, min_samples={min_samples})',
                            n_clusters_found,
                            noise_points
                        )
                        if result is not None:
                            results.append(result)
                    else:
                        print(f"DBSCAN ({metric}, eps={eps}, min_samples={min_samples}) found 0 clusters (only noise). Skipping metrics.")
                except Exception as e:
                    print(f"Error during DBSCAN (metric={metric}, eps={eps}, min_samples={min_samples}): {e}")

    # 4. Spectral Clustering
    for n_clusters in cluster_numbers: # Should be [30]
        try:
            # Using default 'rbf' affinity, can also try 'nearest_neighbors'
            # assign_labels can be 'kmeans' or 'discretize'
            spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=run_seed, affinity='rbf') # Used run_seed
            spectral_labels = spectral.fit_predict(X_scaled) # Changed from X_flat to X_scaled
            result = calculate_metrics(spectral_labels, X_scaled, f'Spectral (rbf, kmeans, n={n_clusters})', n_clusters) # Changed from X_flat to X_scaled
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error during Spectral Clustering (n={n_clusters}): {e}")


    # Analyze top models for correspondence AFTER all results are collected - Commented out for 10-run averaging
    # print(f"\nAnalyzing top models for size={embedding_size}, window={window}...")
    # analyze_top_models(kmeans_results, 'K-means', top_k=2)
    # analyze_top_models(hierarchical_results, 'Hierarchical', top_k=3)
    # analyze_top_models(dbscan_results, 'DBSCAN', top_k=3)
    # analyze_top_models(spectral_results, 'Spectral', top_k=3) 

    
    # print(f"Finished analyzing top models for size={embedding_size}, window={window}. Correspondences collected: {len(correspondences)}") # Commented out

    return pd.DataFrame(results) # Return only results DataFrame for 10-run averaging

# ----- New Function for Parallel Processing -----
def process_embedding_combination(size, window, presence_clusters_df_global, run_seed):
    """Processes a single combination of embedding size and window for a given run seed."""
    print(f"Starting processing for embedding size: {size}, window size: {window}, seed: {run_seed}")
    results_df = pd.DataFrame() 
    # correspondences = {} # Commented out for 10-run averaging
    try:
        # embedding_file = f'/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings_data/filtered_embeddings/filtered_unified_embeddings_aggregated_{size}_{window}_averaged_taste.pkl'
        embedding_file = f'/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/filtered_embeddings/filtered_unified_embeddings_{size}_{window}.pkl'
        # embedding_file = f'/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings_data/filtered_embeddings/filtered_unified_embeddings_aggregated_{size}_{window}_sg0_neg15_ep10.pkl'
        final_df_local = pd.read_pickle(embedding_file)
        print(f"Loaded {final_df_local.shape[0]} embeddings from {embedding_file}")
        
        final_df_local['food_name'] = final_df_local['food_name'].replace('Dragée', 'Dragee')
        final_df_local['food_name'] = final_df_local['food_name'].replace('Cupuaçu', 'Cupuacu')

        if final_df_local.empty:
             print(f"Warning: Embedding file {embedding_file} is empty. Skipping.")
             return results_df 

        X_flat = np.array(final_df_local['unified_embedding'].tolist())
        
        # Raw embeddings  (shape: n_foods × d)
        # X_flat_raw = np.array(final_df_local['unified_embedding'].tolist())

        # # # ④  L2-normalise each row → unit vectors   (cosine ≡ Euclidean here)
        # # X_flat = normalize(X_flat_raw, norm='l2', axis=1)

        # # ----------------------------------------------------------------------------
        # # Replace the *two* lines where you currently build X_flat inside
        # # process_embedding_combination() with the block below
        # # ----------------------------------------------------------------------------
        # PCA_VARIANCE = 0.95   # keep PCs that preserve 95 % total variance

        # # ---------- raw embedding matrix (n_foods × d) ------------------------------
        # # X_raw = np.vstack(final_df_local['unified_embedding'])

        # # ---------- mean-centre ------------------------------------------------------
        # X_center = X_flat_raw - X_flat_raw.mean(axis=0, keepdims=True)
        
        # from sklearn.decomposition import PCA

        # # ---------- PCA dimensionality reduction ------------------------------------
        # pca      = PCA(n_components=PCA_VARIANCE, svd_solver='full', random_state=0)
        # X_pca    = pca.fit_transform(X_center)        # shape: n_foods × d_reduced

        # # ---------- L2-normalise rows → unit vectors --------------------------------
        # X_flat   = normalize(X_pca, norm='l2', axis=1)

        # (optional) keep track of how many dims survived for logging
        # print(f'  PCA kept {X_pca.shape[1]} dims (variance ≥ {PCA_VARIANCE:.2%})')
        
        # ----------------------------------------------------------------------------
        # Evaluate clustering algorithms, get results (correspondences part is handled by commenting within the function)
        results_df = evaluate_clustering_algorithms(X_flat, presence_clusters_df_global, final_df_local, size, window, run_seed) # Pass run_seed
        # print(f"Finished processing for embedding size: {size}, window size: {window}, seed: {run_seed}. Found {len(results_df)} results and {len(correspondences)} correspondences.") # Old print
        print(f"Finished processing for embedding size: {size}, window size: {window}, seed: {run_seed}. Found {len(results_df)} results.")
        
    except FileNotFoundError:
        print(f"Warning: Embedding file not found for size={size}, window={window}. Path: {embedding_file}. Skipping.")
    except Exception as e:
        print(f"Error processing size={size}, window={window}, seed: {run_seed}: {e}")
        
    return results_df # Return only results_df for 10-run averaging

# ----- Main Execution Logic -----
embedding_sizes = [50, 100, 150, 200, 250]
window_sizes = [2, 3, 5, 7, 10]
N_RUNS = 10 # Number of runs for averaging

all_runs_final_results_list = []
# all_correspondences = {} # Commented out for 10-run averaging

for run_i in range(N_RUNS):
    current_run_seed = np.random.randint(0, 100000) # Generate a new seed for this run
    print(f"--- Starting Run {run_i + 1}/{N_RUNS} with seed {current_run_seed} ---")

    print(f"Starting parallel processing for run {run_i + 1} across {len(embedding_sizes) * len(window_sizes)} combinations...")
    # Each element in parallel_outputs_current_run is now just a results_df
    parallel_outputs_current_run = Parallel(n_jobs=-1)( 
        delayed(process_embedding_combination)(size, window, presence_clusters_df, current_run_seed) 
        for size in embedding_sizes 
        for window in window_sizes
    )

    print(f"Parallel processing for run {run_i + 1} finished. Combining results for this run...")

    current_run_results_list = []
    # Iterate through the list of DataFrames returned by parallel processing for the current run
    for results_df_combo in parallel_outputs_current_run:
        if results_df_combo is not None and not results_df_combo.empty:
            current_run_results_list.append(results_df_combo)
    # Correspondence handling is fully commented out
    # if correspondences_dict_combo: 
    #     all_correspondences.update(correspondences_dict_combo)

    if current_run_results_list:
        single_run_final_df = pd.concat(current_run_results_list, ignore_index=True)
        single_run_final_df['run'] = run_i + 1 # Add run number for traceability before averaging
        single_run_final_df['seed'] = current_run_seed # Add seed used for this run
        all_runs_final_results_list.append(single_run_final_df)
        print(f"Run {run_i + 1} completed. {len(single_run_final_df)} results collected for this run.")
    else:
        print(f"No results generated for run {run_i + 1}.")

print("All runs completed. Aggregating final results...")

# Combine all results DataFrames from all runs into one mega DataFrame
if all_runs_final_results_list:
    mega_results_df = pd.concat(all_runs_final_results_list, ignore_index=True)
    
    # Define columns for grouping and aggregation
    grouping_cols = ['embedding_size', 'window_size', 'algorithm', 'n_clusters'] # Added 'n_clusters' to grouping
    # Columns to average
    metric_cols_to_average = [
        'silhouette', 'calinski', 'davies', 'ari', 'nmi', 'homogeneity', 
        'completeness', 'v_measure', 'precision', 'recall', 'f1', 'fmi', 'vi',
        'micro_precision', 'micro_recall', 'micro_f1', 'noise_points',
        'noise_percentage', 'clustered_points', 'clustered_percentage'
    ]
    # Columns that should be constant per group, take the first occurrence
    constant_cols_to_first = ['total_points_in_embedding', 'compared_points']

    agg_dict = {}
    for col in metric_cols_to_average:
        if col in mega_results_df.columns:
            agg_dict[col] = 'mean'
    for col in constant_cols_to_first:
        if col in mega_results_df.columns:
            agg_dict[col] = 'first' 

    if not agg_dict:
        print("Warning: No metric columns found for aggregation. Averaged results will be empty.")
        averaged_final_results = pd.DataFrame()
    else:
        averaged_final_results = mega_results_df.groupby(grouping_cols).agg(agg_dict).reset_index()

else:
    averaged_final_results = pd.DataFrame() 
    print("No results collected across all runs. Averaged results will be empty.")

# --- Post-processing and Saving --- 
# This section now processes and saves the averaged_final_results
if not averaged_final_results.empty:
    print("\nMean Results over 10 runs, sorted by different metrics:")
    metrics_to_print = ['ari', 'nmi', 'v_measure', 'homogeneity', 'completeness', 'precision', 'recall', 'f1', 'vi'] 
    for metric in metrics_to_print:
        if metric in averaged_final_results.columns:
            print(f"\nSorted by {metric} (mean over 10 runs):")
            try:
                pivot_df = averaged_final_results.pivot_table(
                    values=metric,
                    index=['embedding_size', 'window_size'],
                    columns='algorithm',
                    aggfunc='first' # Should be unique after groupby, so 'first' is fine
                )
                print(pivot_df)
            except Exception as e:
                print(f"Could not create pivot table for {metric}: {e}")
        else:
            print(f"Metric '{metric}' not found in averaged results.")

    # Save detailed averaged results
    output_file = "/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clustering_comparison_results/comparison_filtered_flavordb_clusters_txt_VS_all_addition_embeddings_filtered_foods_683_parallel_fullparams_fixCupuaDragee_spectral_multi_clust_avg10runs_new_distance_metrics_xscaled_kmeans.csv" # Updated filename
    averaged_final_results.to_csv(output_file, index=False)
    print(f"\nDetailed averaged results saved to {output_file}. This file contains one row per model configuration (including n_clusters), with metrics averaged over 10 runs.")

    # Correspondence saving is entirely commented out
    # # print(f'\nSaving {len(all_correspondences)} correspondence tables...')
    # # correspondence_output_dir = "/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clustering_comparison_results/correspondence_tables_addition_embeddings_spectral_30clust_avg10runs/" 
    # # import os
    # # os.makedirs(correspondence_output_dir, exist_ok=True) 
    # # 
    # # for name, corr in all_correspondences.items():
    # #     if corr is not None and isinstance(corr, pd.DataFrame):
    # #         try:
    # #             corr_filename = f"correspondence_{name}_flavordb_VS_additionEmbeddings_683_fixCupuaDragee_spectral_30clust_avg10runs.csv" 
    # #             corr_filepath = os.path.join(correspondence_output_dir, corr_filename)
    # #             corr.to_csv(corr_filepath)
    # #             print(f"  Saved: {corr_filename}")
    # #         except Exception as e:
    # #             print(f"  Error saving correspondence table {name}: {e}")
    # #     else:
    # #          print(f"  Skipping invalid correspondence data for {name}.")

else:
    print("\nNo averaged results were generated. Please check input files and logs for each run.") 