import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import re
from joblib import Parallel, delayed
from sklearn.metrics import fowlkes_mallows_score
from bcubed_metrics.bcubed import Bcubed

# Ground truth cluster file
reorganized_clusters_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/filtered_flavordb_clusters_683.txt'
presence_clusters_dict = {}

def fmi_pair_counting(true_labels, pred_labels):
    return fowlkes_mallows_score(true_labels, pred_labels)

def variation_of_information(true, pred):
    h_true = mutual_info_score(true, true)
    h_pred = mutual_info_score(pred, pred)
    mi     = mutual_info_score(true, pred)
    return h_true + h_pred - 2*mi

def split_foods(line):
    return re.split(r',\s*(?![^()]*\))', line)

with open(reorganized_clusters_path, 'r') as file:
    current_cluster = None
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Cluster'):
            current_cluster = line.split('Cluster ')[-1].strip(':')
            presence_clusters_dict[current_cluster] = []
        elif line and current_cluster is not None:
            foods = [food.strip() for food in split_foods(line)]
            presence_clusters_dict[current_cluster].extend(foods)

presence_clusters_list = [(food, cluster) for cluster, foods in presence_clusters_dict.items() for food in foods]
presence_clusters_df_global = pd.DataFrame(presence_clusters_list, columns=["food_name", "presence_cluster"])
print(f"Loaded {presence_clusters_df_global.shape[0]} foods from ground truth cluster file {reorganized_clusters_path}")
print(f"Unique foods in ground truth clusters: {presence_clusters_df_global.food_name.nunique()}")

# def analyze_cluster_correspondence(presence_labels, predicted_labels, food_names):
#     """
#     Analyze how clusters correspond between two clustering methods
#     """
#     df = pd.DataFrame({
#         'food': food_names,
#         'presence': presence_labels,
#         'predicted': predicted_labels
#     })
#     correspondence = pd.crosstab(df['presence'], df['predicted'])
#     correspondence_pct = correspondence.div(correspondence.sum(axis=1), axis=0) * 100
#     print("\nCluster Correspondence (%):")
#     print(correspondence_pct)
#     best_matches = []
#     for presence_cluster in correspondence_pct.index:
#         if presence_cluster in correspondence_pct.index and not correspondence_pct.loc[presence_cluster].empty:
#              best_match = correspondence_pct.loc[presence_cluster].idxmax()
#              overlap = correspondence_pct.loc[presence_cluster, best_match]
#              best_matches.append({
#                  'presence_cluster': presence_cluster,
#                  'predicted_cluster': best_match,
#                  'overlap_percentage': overlap
#              })
#         else:
#              print(f"Warning: Presence cluster '{presence_cluster}' has no corresponding predicted clusters in this subset.")
#     print("\nBest Matching Clusters:")
#     for match in best_matches:
#         print(f"Presence Cluster {match['presence_cluster']} → "
#               f"Predicted Cluster {match['predicted_cluster']} "
#               f"(Overlap: {match['overlap_percentage']:.1f}%)")
#     return correspondence_pct

def clustering_precision_recall_hungarian(true_labels, pred_labels):
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    true_to_idx = {label: idx for idx, label in enumerate(unique_true)}
    pred_to_idx = {label: idx for idx, label in enumerate(unique_pred)}
    true_indices = np.array([true_to_idx[label] for label in true_labels])
    pred_indices = np.array([pred_to_idx[label] for label in pred_labels])
    n_true_clusters = len(unique_true)
    n_pred_clusters = len(unique_pred)
    all_indices = list(range(max(n_true_clusters, n_pred_clusters)))
    cm = confusion_matrix(true_indices, pred_indices, labels=all_indices)
    
    # df = pd.DataFrame({'true': true_labels, 'pred': pred_labels})
    # correspondence_raw = pd.crosstab(df['true'], df['pred'], dropna=False)
    # correspondence_pct_pred = correspondence_raw.div(correspondence_raw.sum(axis=0), axis=1).fillna(0) * 100
    # print("\nCluster Correspondence (% of Predicted Cluster):") # Commented out for brevity
    # print(correspondence_pct_pred) # Commented out for brevity

    cost_matrix = -cm 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    TP = cm[row_ind, col_ind].sum()
    FP = cm[:, col_ind].sum() - TP
    FN = cm[row_ind, :].sum() - TP
    micro_precision = TP / (TP + FP) if TP + FP else 0.0
    micro_recall    = TP / (TP + FN) if TP + FN else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    
    # print(f"\nMicro Precision: {micro_precision:.3f}") # Commented out for brevity
    # print(f"Micro Recall:    {micro_recall:.3f}") # Commented out for brevity
    # print(f"Micro F1:        {micro_f1:.3f}") # Commented out for brevity

    # print("\nOptimal Cluster Matching:") # Commented out to reduce console output for 10 runs
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        if i < n_true_clusters and j < n_pred_clusters:
            total_true = cm[i, :].sum()
            total_pred = cm[:, j].sum()
            if total_true > 0 and total_pred > 0: 
                 matched_pairs.append((i, j))

    precision_per_cluster = []
    recall_per_cluster = []
    for i, j in matched_pairs:
        total_pred = cm[:, j].sum()
        total_true = cm[i, :].sum()
        precision = cm[i, j] / total_pred
        recall = cm[i, j] / total_true
        precision_per_cluster.append(precision)
        recall_per_cluster.append(recall)

    macro_precision = np.mean(precision_per_cluster) if precision_per_cluster else 0
    macro_recall = np.mean(recall_per_cluster) if recall_per_cluster else 0
    f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if macro_precision + macro_recall > 0 else 0.0
    
    # print(f"\nOverall Metrics (based on {len(matched_pairs)} matched pairs):") # Commented out for brevity
    # print(f"Macro Precision: {macro_precision:.3f}") # Commented out for brevity
    # print(f"Macro Recall: {macro_recall:.3f}") # Commented out for brevity
    # print(f"F1 Score: {f1_score:.3f}") # Commented out for brevity

    return macro_precision, macro_recall, f1_score, micro_precision, micro_recall, micro_f1

def evaluate_clustering_algorithms(X_flat_presence, food_names_presence, run_seed):
    results = []
    scaler = StandardScaler()
    X_scaled_presence = scaler.fit_transform(X_flat_presence)
    
    # Construct final_df_local for metric calculation, based on the food names from the presence dataset
    final_df_local_metric = pd.DataFrame({'food_name': food_names_presence})

    def calculate_metrics(labels, X_data, algorithm_name, n_clusters_metric, noise_points=0):
        total_points = len(X_data)
        silhouette, calinski, davies = None, None, None
        try:
            if 'DBSCAN' in algorithm_name and noise_points > 0:
                valid_mask = labels != -1
                if np.sum(valid_mask) > 1:
                     valid_X = X_data[valid_mask]
                     valid_labels = labels[valid_mask]
                     if len(set(valid_labels)) >= 2:
                         silhouette = silhouette_score(valid_X, valid_labels)
                         calinski = calinski_harabasz_score(valid_X, valid_labels)
                         davies = davies_bouldin_score(valid_X, valid_labels)
            elif len(set(labels)) - (1 if -1 in labels else 0) >= 2:
                 silhouette = silhouette_score(X_data, labels)
                 calinski = calinski_harabasz_score(X_data, labels)
                 davies = davies_bouldin_score(X_data, labels)
        except ValueError as e:
             print(f"Could not calculate internal metrics for {algorithm_name}: {e}")
        except Exception as e:
             print(f"An unexpected error during internal metric calculation for {algorithm_name}: {e}")
        
        temp_df_metric = final_df_local_metric.copy()
        temp_df_metric['cluster'] = labels
        comparison_df_merged = pd.merge(presence_clusters_df_global, temp_df_metric, on='food_name', how='inner')
        
        ari, nmi, homogeneity, completeness, v_measure, precision, recall, f1, micro_precision, micro_recall, micro_f1, fmi_val, vi_val = (None,) * 13

        if comparison_df_merged.empty or comparison_df_merged['presence_cluster'].nunique() < 1 or comparison_df_merged['cluster'].nunique() < 1:
            print(f"Warning: Insufficient data for supervised metrics for {algorithm_name}. Skipping.")
        else:
            true_labels_common = comparison_df_merged['presence_cluster']
            pred_labels_common = comparison_df_merged['cluster']
            try:
                ari = adjusted_rand_score(true_labels_common, pred_labels_common)
                nmi = normalized_mutual_info_score(true_labels_common, pred_labels_common)
                homogeneity = homogeneity_score(true_labels_common, pred_labels_common)
                completeness = completeness_score(true_labels_common, pred_labels_common)
                v_measure = v_measure_score(true_labels_common, pred_labels_common)
                precision, recall, f1, micro_precision, micro_recall, micro_f1 = clustering_precision_recall_hungarian(
                    true_labels_common, pred_labels_common
                )
                fmi_val = fmi_pair_counting(true_labels_common, pred_labels_common)
                vi_val = variation_of_information(true_labels_common, pred_labels_common)
            except ValueError as e:
                print(f"Could not calculate supervised metrics for {algorithm_name}: {e}")
            except Exception as e:
                 print(f"An unexpected error occurred during supervised metric calculation for {algorithm_name}: {e}")
        
        result = {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters_metric, # This is n_clusters for K-Means/Hierarchical/Spectral, or n_clusters_found for DBSCAN
            'noise_points': noise_points if noise_points else 0,
            'silhouette': silhouette, 'calinski': calinski, 'davies': davies,
            'ari': ari, 'nmi': nmi, 'homogeneity': homogeneity, 'completeness': completeness, 'v_measure': v_measure,
            'precision': precision, 'recall': recall, 'f1': f1, 'fmi': fmi_val, 'vi': vi_val,
            'micro_precision': micro_precision, 'micro_recall': micro_recall, 'micro_f1': micro_f1,
            'total_points_in_dataset': total_points, 
            'compared_points': len(comparison_df_merged), 
            'noise_percentage': (noise_points / total_points) * 100 if total_points > 0 else 0,
            'clustered_points': total_points - noise_points,
            'clustered_percentage': ((total_points - noise_points) / total_points) * 100 if total_points > 0 else 0
        }
        return result

    # cluster_numbers_list = [10, 15, 20, 25, 30, 35]
    cluster_numbers_list = [30]
    
    # 1. K-means
    for n_clusters_iter in cluster_numbers_list:
        try:
            kmeans = KMeans(n_clusters=n_clusters_iter, n_init=50, init='k-means++', random_state=run_seed)
            kmeans_labels = kmeans.fit_predict(X_flat_presence)
            metric_values = calculate_metrics(kmeans_labels, X_flat_presence, f'K-means++ (n={n_clusters_iter})', n_clusters_iter)
            if metric_values is not None: results.append(metric_values)
        except Exception as e: print(f"Error during K-means (n={n_clusters_iter}): {e}")

    # 2. Hierarchical Clustering
    for n_clusters_iter in cluster_numbers_list:
        for link in ['ward', 'complete', 'average', 'single']: 
            # 'ward' typically requires euclidean metric. Presence data is binary, Jaccard often preferred.
            # For simplicity, if 'ward', use X_flat_presence as is, it might not be optimal without conversion/scaling if data isn't euclidean-like.
            # Scipy's linkage (used in check_type_linkage.py) is more flexible with metrics directly.
            # AgglomerativeClustering default metric is 'euclidean'.
            # To use Jaccard with Agglomerative, precompute distance matrix or ensure data is suitable for euclidean if not.
            # Here, we proceed with default 'euclidean' for Agglomerative for simplicity matching the spectral script.
            # Consider if Jaccard or other binary metrics are needed and how to integrate with AgglomerativeClustering.
            try:
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters_iter, linkage=link)
                hierarchical_labels = hierarchical.fit_predict(X_flat_presence)
                metric_values = calculate_metrics(hierarchical_labels, X_flat_presence, f'Hierarchical ({link}, n={n_clusters_iter})', n_clusters_iter)
                if metric_values is not None: results.append(metric_values)
            except Exception as e: print(f"Error during Hierarchical (linkage={link}, n={n_clusters_iter}): {e}")

    # 3. DBSCAN
    eps_values = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] 
    min_samples_values = [2, 3, 4, 5] 
    db_metrics = ['euclidean', 'cosine'] # Jaccard could be used if X_flat_presence is suitable or precomputed distance matrix
    
    for db_metric_name in db_metrics:
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    X_use_dbscan = X_scaled_presence if db_metric_name == 'euclidean' else X_flat_presence
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=db_metric_name)
                    dbscan_labels = dbscan.fit_predict(X_use_dbscan)
                    n_clusters_found_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                    noise_points_db = sum(1 for x in dbscan_labels if x == -1)
                    if n_clusters_found_db >= 1: 
                        metric_values = calculate_metrics(dbscan_labels, X_use_dbscan, f'DBSCAN ({db_metric_name}, eps={eps}, min_samples={min_samples})', n_clusters_found_db, noise_points_db)
                        if metric_values is not None: results.append(metric_values)
                    # else: print(f"DBSCAN ({db_metric_name}, eps={eps}, min_samples={min_samples}) found 0 clusters. Skipping.")
                except Exception as e: print(f"Error during DBSCAN (metric={db_metric_name}, eps={eps}, min_samples={min_samples}): {e}")

    # 4. Spectral Clustering
    for n_clusters_iter in cluster_numbers_list:
        try:
            spectral = SpectralClustering(n_clusters=n_clusters_iter, assign_labels='kmeans', random_state=run_seed, affinity='rbf')
            spectral_labels = spectral.fit_predict(X_flat_presence) # Using X_flat_presence; affinity='rbf' uses euclidean distances. Consider 'precomputed' if using Jaccard.
            metric_values = calculate_metrics(spectral_labels, X_flat_presence, f'Spectral (rbf, kmeans, n={n_clusters_iter})', n_clusters_iter)
            if metric_values is not None: results.append(metric_values)
        except Exception as e: print(f"Error during Spectral Clustering (n={n_clusters_iter}): {e}")

    return pd.DataFrame(results)

# ----- Main Execution Logic -----
N_RUNS = 10
all_runs_results_list = []

# Load the presence dataset
presence_data_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/compounds_presence/foodname_compound_presence_0_1_filtered.csv'
df_presence_input = pd.read_csv(presence_data_path, sep=';', index_col=0)
X_presence_data = df_presence_input.values # Get numpy array
food_names_from_presence = df_presence_input.index.tolist() # Get food names

print(f"Loaded presence data from {presence_data_path} with shape {X_presence_data.shape}")

# Generate seeds for each run beforehand
seeds_for_runs = [np.random.randint(0, 100000) for _ in range(N_RUNS)]

print(f"Starting {N_RUNS} runs in parallel...")

# Use joblib to run iterations in parallel
# Each call to evaluate_clustering_algorithms is one entire run for a given seed.
parallel_outputs = Parallel(n_jobs=-1)(
    delayed(evaluate_clustering_algorithms)(X_presence_data, food_names_from_presence, seeds_for_runs[i]) 
    for i in range(N_RUNS)
)

print("Parallel runs finished. Processing results...")

# Process results from parallel runs
all_runs_results_list = []
for i, run_results_df in enumerate(parallel_outputs):
    current_seed_used = seeds_for_runs[i]
    run_number = i + 1
    print(f"Processing results from run {run_number} (seed: {current_seed_used})")
    if run_results_df is not None and not run_results_df.empty:
        run_results_df['run'] = run_number
        run_results_df['seed'] = current_seed_used
        all_runs_results_list.append(run_results_df)
        print(f"Run {run_number} completed. {len(run_results_df)} algorithm configurations evaluated.")
    else:
        print(f"No results generated for run {run_number} (seed: {current_seed_used}).")


print("All runs completed and processed. Aggregating final results...")

if all_runs_results_list:
    mega_results_df = pd.concat(all_runs_results_list, ignore_index=True)
    
    grouping_cols = ['algorithm', 'n_clusters'] # n_clusters is part of algorithm name for DBSCAN
    
    metric_cols_to_average = [
        'silhouette', 'calinski', 'davies', 'ari', 'nmi', 'homogeneity', 
        'completeness', 'v_measure', 'precision', 'recall', 'f1', 'fmi', 'vi',
        'micro_precision', 'micro_recall', 'micro_f1', 'noise_points',
        'noise_percentage', 'clustered_points', 'clustered_percentage'
    ]
    constant_cols_to_first = ['total_points_in_dataset', 'compared_points'] # n_clusters is handled by grouping_cols for non-DBSCAN or within algorithm name str for DBSCAN

    agg_dict = {}
    for col in metric_cols_to_average:
        if col in mega_results_df.columns: agg_dict[col] = 'mean'
    for col in constant_cols_to_first:
        if col in mega_results_df.columns: agg_dict[col] = 'first' 

    if not agg_dict:
        print("Warning: No metric columns found for aggregation.")
        averaged_final_results = pd.DataFrame()
    else:
        # For DBSCAN, 'n_clusters' is n_clusters_found and part of 'algorithm' name.
        # For others, 'n_clusters' is a distinct column.
        # We group by 'algorithm' (which contains DBSCAN params) and 'n_clusters' (which is set for others).
        # Rows where 'n_clusters' is NaN (like DBSCAN summary rows if they existed) might behave differently.
        # The calculate_metrics ensures 'n_clusters' is n_clusters_found for DBSCAN, so it should be fine.
        averaged_final_results = mega_results_df.groupby(grouping_cols, dropna=False).agg(agg_dict).reset_index()
else:
    averaged_final_results = pd.DataFrame() 
    print("No results collected across all runs.")

if not averaged_final_results.empty:
    print("\nMean Results over 10 runs, sorted by different metrics:")
    metrics_to_print = ['ari', 'nmi', 'v_measure', 'homogeneity', 'completeness', 'precision', 'recall', 'f1', 'vi'] 
    for metric in metrics_to_print:
        if metric in averaged_final_results.columns:
            print(f"\nSorted by {metric} (mean over 10 runs):")
            try:
                # Pivoting might be tricky if 'n_clusters' column has mixed meanings or NaNs due to DBSCAN
                # For now, print the grouped table directly
                print(averaged_final_results[['algorithm', 'n_clusters', metric]].sort_values(by=metric, ascending=False if metric not in ['davies'] else True))
            except Exception as e:
                print(f"Could not print sorted table for {metric}: {e}")
        else:
            print(f"Metric '{metric}' not found in averaged results.")

    output_file = "/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clustering_comparison_results/presence_dataset_clustering_comparison_avg10runs_30clusters.csv" 
    averaged_final_results.to_csv(output_file, index=False)
    print(f"\nDetailed averaged results saved to {output_file}.")
else:
    print("\nNo averaged results were generated.") 