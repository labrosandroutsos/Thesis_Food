import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import sys
import os

def find_optimal_k(embedding_file_path, max_k=50, k_step=1):
    """
    Calculates and plots inertia and silhouette scores for K-Means
    across a range of K values to help find the optimal K.

    Args:
        embedding_file_path (str): Path to the .pkl file containing embeddings.
                                   Expected format: DataFrame with 'food_name' and 'unified_embedding' columns.
        max_k (int): The maximum number of clusters (K) to test.
        k_step (int): The step size for K values (e.g., 1 to test 2, 3, 4..., 2 to test 2, 4, 6...).
    """
    if not os.path.exists(embedding_file_path):
        print(f"Error: Embedding file not found at {embedding_file_path}")
        return

    print(f"Loading embeddings from: {embedding_file_path}")
    try:
        df = pd.read_pickle(embedding_file_path)
        if 'unified_embedding' not in df.columns:
            print("Error: 'unified_embedding' column not found in the DataFrame.")
            return
        X = np.array(df['unified_embedding'].tolist())
        if X.ndim != 2 or X.shape[1] == 0:
             print("Error: Embedding data is not in the expected 2D array format.")
             return
        print(f"Loaded data shape: {X.shape}")
    except Exception as e:
        print(f"Error loading or processing embedding file: {e}")
        return

    k_values = range(2, max_k + 1, k_step) # K must be >= 2 for silhouette
    inertias = []
    silhouette_scores = []

    print(f"Calculating metrics for K from 2 to {max_k} (step={k_step})...")
    for k in k_values:
        try:
            print(f"  Processing K={k}...")
            kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k >= 2:
                 score = silhouette_score(X, kmeans.labels_)
                 silhouette_scores.append(score)
                 print(f"    K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
            else:
                 silhouette_scores.append(np.nan)
                 print(f"    K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette=N/A")

        except Exception as e:
            print(f"Error during K-Means for K={k}: {e}")
            inertias.append(np.nan)
            silhouette_scores.append(np.nan)

    # Find best K based on silhouette score
    valid_silhouette_indices = [i for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
    if valid_silhouette_indices:
        best_silhouette_index = valid_silhouette_indices[np.argmax([silhouette_scores[i] for i in valid_silhouette_indices])]
        best_k_silhouette = k_values[best_silhouette_index]
        print(f"\nOptimal K based on Silhouette Score: {best_k_silhouette} (Score: {silhouette_scores[best_silhouette_index]:.4f})")
    else:
        print("\nCould not determine optimal K from Silhouette Score (no valid scores).")
        best_k_silhouette = None

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of clusters (K)')
    ax1.set_ylabel('Inertia (WCSS)', color=color)
    ax1.plot(k_values, inertias, marker='o', linestyle='-', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Elbow Method and Silhouette Score for Optimal K')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_values, silhouette_scores, marker='x', linestyle='--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    if best_k_silhouette is not None:
         ax2.axvline(x=best_k_silhouette, color='grey', linestyle=':', linewidth=1.5, label=f'Best Silhouette K={best_k_silhouette}')
         ax1.axvline(x=best_k_silhouette, color='grey', linestyle=':', linewidth=1.5)

    fig.tight_layout()
    fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5), bbox_transform=ax1.transAxes)
    
    plot_filename = os.path.splitext(os.path.basename(embedding_file_path))[0] + '_optimal_k.png'
    plot_save_path = os.path.join(os.path.dirname(embedding_file_path), plot_filename)
    try:
        plt.savefig(plot_save_path)
        print(f"Plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Optionally show plot interactively

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_optimal_k.py <path_to_embedding_pickle_file> [max_k] [k_step]")
        sys.exit(1)
        
    embedding_path = sys.argv[1]
    max_k_arg = 50
    k_step_arg = 1
    
    if len(sys.argv) > 2:
        try:
            max_k_arg = int(sys.argv[2])
        except ValueError:
            print("Warning: Invalid max_k provided. Using default 50.")
            
    if len(sys.argv) > 3:
        try:
            k_step_arg = int(sys.argv[3])
            if k_step_arg < 1:
                 print("Warning: k_step must be >= 1. Using default 1.")
                 k_step_arg = 1
        except ValueError:
            print("Warning: Invalid k_step provided. Using default 1.")

    find_optimal_k(embedding_path, max_k=max_k_arg, k_step=k_step_arg) 