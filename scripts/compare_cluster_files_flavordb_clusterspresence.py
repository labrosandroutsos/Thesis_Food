import pandas as pd
import numpy as np
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                           homogeneity_score, completeness_score, v_measure_score,
                           silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import re

def split_foods(line):
    """Use a regular expression to split on commas that are not within parentheses"""
    return re.split(r',\s*(?![^()]*\))', line)

def load_clusters_from_file(filepath):
    """Load clusters from a txt file into a dictionary"""
    clusters_dict = {}
    with open(filepath, 'r') as file:
        current_cluster = None
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if line.startswith('Cluster'):
                current_cluster = line.split('Cluster ')[-1].strip(':')
                clusters_dict[current_cluster] = []
            elif line and current_cluster is not None:
                foods = [food.strip() for food in split_foods(line)]
                clusters_dict[current_cluster].extend(foods)
    return clusters_dict

def create_clusters_df(clusters_dict):
    """Convert clusters dictionary to DataFrame"""
    clusters_list = [(food, cluster) for cluster, foods in clusters_dict.items() for food in foods]
    return pd.DataFrame(clusters_list, columns=["food_name", "cluster"])

def analyze_cluster_correspondence(cluster1_labels, cluster2_labels, food_names):
    """Analyze how clusters correspond between two clustering methods"""
    df = pd.DataFrame({
        'food': food_names,
        'cluster1': cluster1_labels,
        'cluster2': cluster2_labels
    })
    
    correspondence = pd.crosstab(df['cluster1'], df['cluster2'])
    correspondence_pct = correspondence.div(correspondence.sum(axis=1), axis=0) * 100
    
    print("\nCluster Correspondence (%):")
    print(correspondence_pct)
    
    best_matches = []
    for cluster1 in correspondence_pct.index:
        best_match = correspondence_pct.loc[cluster1].idxmax()
        overlap = correspondence_pct.loc[cluster1, best_match]
        best_matches.append({
            'cluster1': cluster1,
            'cluster2': best_match,
            'overlap_percentage': overlap
        })
    
    print("\nBest Matching Clusters:")
    for match in best_matches:
        print(f"Cluster 1: {match['cluster1']} → "
              f"Cluster 2: {match['cluster2']} "
              f"(Overlap: {match['overlap_percentage']:.1f}%)")
    
    return correspondence_pct


def clustering_precision_recall(true_labels, pred_labels):
    """Calculate precision, recall, and F1 score for clustering results"""
    df = pd.DataFrame({
        'true': true_labels,
        'pred': pred_labels
    })
    correspondence = pd.crosstab(df['true'], df['pred'])
    
    precision_per_cluster = []
    recall_per_cluster = []
    
    for true_cluster in correspondence.index:
        best_pred_cluster = correspondence.loc[true_cluster].idxmax()
        precision = correspondence.loc[true_cluster, best_pred_cluster] / correspondence[best_pred_cluster].sum()
        recall = correspondence.loc[true_cluster, best_pred_cluster] / correspondence.loc[true_cluster].sum()
        precision_per_cluster.append(precision)
        recall_per_cluster.append(recall)
    
    macro_precision = np.mean(precision_per_cluster)
    macro_recall = np.mean(recall_per_cluster)
    
    f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if macro_precision + macro_recall > 0 else 0.0
    
    return macro_precision, macro_recall, f1_score

def clustering_precision_recall_hungarian_same_with_embeddings_script(true_labels, pred_labels):
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
    print("\nCluster Correspondence (Raw Counts):")
    print(correspondence_raw)
    
    # Calculate and print percentage overlap from true cluster perspective
    correspondence_pct_true = correspondence_raw.div(correspondence_raw.sum(axis=1), axis=0).fillna(0) * 100
    print("\nCluster Correspondence (% of True Cluster):")
    print(correspondence_pct_true)
    
    # Calculate and print percentage overlap from predicted cluster perspective
    correspondence_pct_pred = correspondence_raw.div(correspondence_raw.sum(axis=0), axis=1).fillna(0) * 100
    print("\nCluster Correspondence (% of Predicted Cluster):")
    print(correspondence_pct_pred)

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

    print(f"\nMicro Precision: {micro_precision:.3f}")
    print(f"Micro Recall:    {micro_recall:.3f}")
    print(f"Micro F1:        {micro_f1:.3f}")

    # Print the optimal matching
    print("\nOptimal Cluster Matching:")
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
                 print(f"True Cluster '{true_cluster}' matched with Predicted Cluster '{pred_cluster}'")
                 print(f"  Overlap: {overlap} items")
                 print(f"  Coverage: {overlap/total_true*100:.1f}% of true cluster, {overlap/total_pred*100:.1f}% of predicted cluster")
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
    
    print(f"\nOverall Metrics (based on {len(matched_pairs)} matched pairs):")
    print(f"Macro Precision: {macro_precision:.3f}")
    print(f"Macro Recall: {macro_recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")

    return macro_precision, macro_recall, f1_score, micro_precision, micro_recall, micro_f1

def clustering_precision_recall_hungarian(true_labels, pred_labels):
    """
    Calculate precision, recall, and F1 score for clustering results using the Hungarian algorithm,
    handling different cluster orderings and labels.
    """
    # Create mapping dictionaries
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    
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

def compare_cluster_files(cluster1_path, cluster2_path):
    """
    Compare two clustering files and calculate various metrics.
    
    Parameters:
    -----------
    cluster1_path : str
        Path to first clustering file
    cluster2_path : str
        Path to second clustering file
    embeddings_path : str, optional
        Path to embeddings file for additional metrics calculation
    embedding_size : int, optional
        Size of embeddings to use if embeddings_path is provided
        
    Returns:
    --------
    dict
        Dictionary containing all calculated metrics
    pd.DataFrame
        Correspondence matrix between the two clusterings
    """
    # Load clusters from files
    clusters1_dict = load_clusters_from_file(cluster1_path)
    clusters2_dict = load_clusters_from_file(cluster2_path)
    
    # Convert to DataFrames
    clusters1_df = create_clusters_df(clusters1_dict)
    clusters2_df = create_clusters_df(clusters2_dict)
    
    print("\nBefore merge:")
    print(f"Cluster1 shape: {clusters1_df.shape}")
    print(f"Cluster2 shape: {clusters2_df.shape}")
    unique_clusters1 = clusters1_df['cluster'].nunique()
    unique_clusters2 = clusters2_df['cluster'].nunique()
    print(f"Number of unique clusters in cluster1: {unique_clusters1}")
    print(f"Number of unique clusters in cluster2: {unique_clusters2}")

    # Print foods that are in cluster1 but not in cluster2
    foods_only_in_1 = set(clusters1_df['food_name']) - set(clusters2_df['food_name'])
    foods_only_in_2 = set(clusters2_df['food_name']) - set(clusters1_df['food_name'])
    
    print(f"\nFoods only in cluster1: {len(foods_only_in_1)}")
    if len(foods_only_in_1) > 0:
        print("Sample foods only in cluster1:", list(foods_only_in_1)[:5])
    
    print(f"Foods only in cluster2: {len(foods_only_in_2)}")
    if len(foods_only_in_2) > 0:
        print("Sample foods only in cluster2:", list(foods_only_in_2)[:5])

    # Merge the clusterings
    comparison_df = pd.merge(
        clusters1_df.rename(columns={'cluster': 'cluster1'}),
        clusters2_df.rename(columns={'cluster': 'cluster2'}),
        on='food_name',
        how='inner'
    )
    
    print("\nAfter merge:")
    print(f"Merged DataFrame shape: {comparison_df.shape}")
    
    # Count foods per cluster before and after merge for cluster1
    print("\nCluster1 foods per cluster:")
    before_merge1 = clusters1_df.groupby('cluster').size()
    after_merge1 = comparison_df.groupby('cluster1').size()
    cluster1_comparison = pd.DataFrame({
        'before_merge': before_merge1,
        'after_merge': after_merge1
    }).fillna(0)
    print(cluster1_comparison)
    
    print("\nCluster2 foods per cluster:")
    before_merge2 = clusters2_df.groupby('cluster').size()
    after_merge2 = comparison_df.groupby('cluster2').size()
    cluster2_comparison = pd.DataFrame({
        'before_merge': before_merge2,
        'after_merge': after_merge2
    }).fillna(0)
    print(cluster2_comparison)

    # how many unique clusters?
    unique_clusters1 = comparison_df['cluster1'].nunique()
    unique_clusters2 = comparison_df['cluster2'].nunique()
    print(f"\nAfter merge - Number of unique clusters in cluster1: {unique_clusters1}")
    print(f"After merge - Number of unique clusters in cluster2: {unique_clusters2}")
    
    # Calculate basic metrics
    metrics = {
        'ari_score': adjusted_rand_score(comparison_df['cluster1'], comparison_df['cluster2']),
        'nmi_score': normalized_mutual_info_score(comparison_df['cluster1'], comparison_df['cluster2']),
        'homogeneity': homogeneity_score(comparison_df['cluster1'], comparison_df['cluster2']),
        'completeness': completeness_score(comparison_df['cluster1'], comparison_df['cluster2']),
        'v_measure': v_measure_score(comparison_df['cluster1'], comparison_df['cluster2'])
    }
    
    # Calculate precision and recall
    precision, recall, f1 = clustering_precision_recall(
        comparison_df['cluster1'],
        comparison_df['cluster2']
    )
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    # Calculate precision and recall using Hungarian algorithm
    precision_hungarian, recall_hungarian, f1_hungarian, micro_precision, micro_recall, micro_f1 = clustering_precision_recall_hungarian_same_with_embeddings_script(
        comparison_df['cluster1'],
        comparison_df['cluster2']
    )
    metrics.update({
        'precision_hungarian': precision_hungarian,
        'recall_hungarian': recall_hungarian,
        'f1_score_hungarian': f1_hungarian,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    })
    
    # # Calculate additional metrics if embeddings are provided
    # if embeddings_path:
    #     final_df = pd.read_pickle(embeddings_path)
    #     X_flat = np.array(final_df['unified_embedding'].tolist())
        
    #     # Merge with embeddings to get the same order of samples
    #     merged_df = pd.merge(comparison_df, final_df[['food_name']], on='food_name', how='inner')
        
    try:
        metrics.update({
            'silhouette_score_cluster': silhouette_score(comparison_df['cluster2'], comparison_df['cluster1']),
            'calinski_harabasz_score_cluster': calinski_harabasz_score(comparison_df['cluster2'], comparison_df['cluster1']),
            'davies_bouldin_score_cluster': davies_bouldin_score(comparison_df['cluster2'], comparison_df['cluster1']),
        })
    except:
        print("Warning: Could not calculate some metrics due to cluster configuration")
    
    # Calculate and display cluster correspondence
    print("\nAnalyzing cluster correspondence:")
    correspondence = analyze_cluster_correspondence(
        comparison_df['cluster1'],
        comparison_df['cluster2'],
        comparison_df['food_name']
    )
    
    # Print metrics
    print("\nMetrics between the two clusterings:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")
    
    # Save results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("cluster_comparison_metrics_683flavordb_vs_foodb_initial_single_correct_linkage_clusters_jaccard_micro.csv", index=False)
    correspondence.to_csv("cluster_correspondence_683flavordb_vs_foodb_initial_single_correct_linkage_clusters_jaccard_micro.csv")
    
    return metrics, correspondence

if __name__ == "__main__":
    # Example usage:
    metrics, correspondence = compare_cluster_files(
        # cluster1_path='/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/filtered_flavordb_clusters.txt',  # First clustering file
        cluster1_path='/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clusters/flavordb_clusters/filtered_flavordb_clusters_683.txt',  # 683 foods clusters
        # cluster1_path='C:/Users/labro/Downloads/final_fully_coherent_clusters.txt',  # First clustering file
        # cluster2_path='compounds_presence/average_linkage_clusters.txt'  # Second clustering file
        cluster2_path='compounds_presence/single_correct_linkage_clusters_distance_30_clusters_jaccard.txt'  # Second clustering file
        # cluster2_path='/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/misc_clusters/average_linkage_clusters_reclustered_gemini_683.txt'  # Second clustering file
        # cluster2_path='/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/misc_clusters/refined_food_clusters_683_gemini_noMisc.txt'  # Second clustering file
        # cluster2_path="processed_flavordb_clusters.txt"
    )
    # clusters_path = 'C:/Users/labro/Downloads/average_linkage_clusters_reclusteredwithKnowledge_24_11'