# Food Compound Analysis and Clustering

This repository contains code and data for analyzing food compounds, generating embeddings, and clustering foods based on their compound profiles. The project focuses on understanding the relationship between food compounds and taste categories (bitter, sweet, umami, and other).

## Project Structure

### Core Directories

- **scripts/**: Contains Python scripts for data processing, embedding generation, and clustering.
  - `autoencoder_try.py`: Experimental script for autoencoders.
  - `apply_filter_to_flavordb_clusters.py`: Applies filters to FlavorDB clusters.
  - `categories_to_cluster.py`: Script for managing categories to cluster.
  - `check_foods_flavordb_foodb.py`: Compares foods between FlavorDB and FooDB databases.
  - `check_foods_number.py`: Checks the number of foods.
  - `check_foods.py`: Script for checking foods.
  - `check_type_linkage.py`: Checks linkage types in clustering.
  - `cluster_embeddings.py`: Applies clustering algorithms (K-means, DBSCAN) to food embeddings.
  - `compare_cluster_files_flavordb_clusterspresence.py`: Compares cluster files (presence) with FlavorDB cluster presence.
  - `compare_clusters_embedding_presence_pca.py`: Compares clustering approaches using embedding presence and PCA.
  - `compare_clusters_embedding_presence.py`: Compares different clustering approaches based on embeddings.
  - `delete_column_compounds.py`: Deletes columns from compound files.
  - `embeddings_transformers.py`: Generates embeddings using transformer models.
  - `fasttext_embeddings.py`: Generates FastText embeddings for compounds.
  - `filter_clusters_by_source_file.py`: Filters clusters based on a source file.
  - `filter_flavordb_foods_683.py`: Filters FlavorDB foods (specific count).
  - `find_optimal_k.py`: Finds the optimal k for K-means clustering.
  - `foods_compounds_0_1.py`: Creates binary presence matrices for compounds in foods.
  - `generate_unified_embeddings.py`: Combines embeddings from different taste categories.
  - `pairwise_things.py`: Script for pairwise comparisons or operations.
  - `pytorch_try.py`: Experimental script using PyTorch.
  - `reassign_clusters.py`: Reassigns clusters based on certain criteria.
  - `remove_foods_from_embeddings.py`: Removes specific foods from embedding sets.
  - `results_clustering.py`: Script for processing or displaying clustering results.
  - `thesis_preprocess_foodb.py`: Preprocesses the FooDB data for the thesis.
  - `unified_embeddings.py`: Works with unified embeddings.
  - `visualize_clusters.py`: Creates visualizations of clusters using PCA and UMAP.
  - `word_embeddings_all.py`: Generates Word2Vec embeddings for all compounds.
  - `word_embeddings.py`: Generates Word2Vec embeddings for compounds across different taste categories.
  - Other utility scripts for data processing and analysis.

- **compounds/**: Contains compound data categorized by taste.
  - `compounds_bitter.csv`: Bitter compounds data (3.2MB)
  - `compounds_sweet.csv`: Sweet compounds data (3.2MB)
  - `compounds_umami.csv`: Umami compounds data (3.2MB)
  - `compounds_other.csv`: Other compounds data (3.2MB)

- **ordered_compounds/**: Contains ordered compounds data per food for each taste category.
  - `ordered_compounds_per_food_bitter.csv`: Ordered bitter compounds per food (131MB)
  - `ordered_compounds_per_food_sweet.csv`: Ordered sweet compounds per food (131MB)
  - `ordered_compounds_per_food_umami.csv`: Ordered umami compounds per food (131MB)
  - `ordered_compounds_per_food_other.csv`: Ordered other compounds per food (131MB)
  - `food_compound_foodb_grouped_bitter_dict.csv`: Grouped bitter compounds dictionary from FooDB (206MB)
  - `food_compound_foodb_grouped_sweet_dict.csv`: Grouped sweet compounds dictionary from FooDB (207MB)
  - `food_compound_foodb_grouped_umami_dict.csv`: Grouped umami compounds dictionary from FooDB (204MB)
  - `food_compound_foodb_grouped_other_dict.csv`: Grouped other compounds dictionary from FooDB (206MB)

### Data and Results Directories

- **embeddings/**: Contains generated embeddings for foods and compounds.
  - `final_unified_embeddings_*.pkl`: Various unified food embeddings (e.g., `final_unified_embeddings_250_10.pkl` (7.4MB)). Files vary by dimensions (50-250) and window sizes (2-10).
  - `num_compounds_foods.csv`: CSV file possibly detailing compound counts per food.
  - `filtered_embeddings/`: Subdirectory for filtered embeddings.
  - `archive/`: Subdirectory likely containing older or archived embedding files.
  - Embeddings are generated using different methods and parameters.

- **embeddings_data/**: Contains processed embedding data and latent representations.
  - `final_unified_embeddings_aggregated_*.pkl`: Aggregated unified embeddings (e.g., `final_unified_embeddings_aggregated_50_2_sg0_neg15_ep10.pkl`).
  - `final_unified_embeddings_fasttext_*.pkl`: Unified embeddings generated using FastText.
  - `X_latent_dataframe.csv` & `X_latent_dataframe_fulldata.csv`: Latent dataframes from dimensionality reduction.
  - `filtered_embeddings/`: Subdirectory for filtered embeddings.
  - Contains various unified embeddings in pickle format, reflecting different aggregation strategies and source models.

- **models/**: Contains trained Word2Vec and FastText models.
  - `word2vec_*.model`: Word2Vec models for different tastes (bitter, sweet, umami, other) with varying dimensions and parameters (e.g., `word2vec_bitter_250_10_sg0_neg15_ep10.model`). Associated `.npy` files are also present.
  - `fasttext_*.model`: FastText models for different tastes with varying parameters (e.g., `fasttext_bitter_50_2_fasttext_sg0_neg5_ep5.model`). Associated `.model.wv.vectors_ngrams.npy` files are present.
  - Models cover different embedding dimensions, window sizes, and training parameters.

- **foodb/**: Contains data from the FooDB database and derived datasets.
  - `FoodDB.csv`: Main FooDB database (65MB).
  - `food_compound_foodb.csv`: Food-compound relationships from FooDB (195MB).
  - `food_compound_foodb_grouped.csv`: Grouped food-compound data (487MB).
  - `food_compound_foodb_grouped_bitter.csv` (243MB), `food_compound_foodb_grouped_sweet.csv` (243MB), `food_compound_foodb_grouped_umami.csv` (240MB), `food_compound_foodb_grouped_other.csv` (242MB): Grouped food-compound data separated by taste.
  - `FoodDB_compounds_and_predictions.csv` (13MB): FooDB compounds with predictions.
  - `FoodDB_compoundnames_and_predictions_only.csv` (6.8MB): Compound names and predictions only.
  - `FooDB_predictions.txt` (6.8MB): Predictions from FooDB.
  - `foods_compounds.csv` (285MB): A dataset of foods and their compounds.
  - `foods_compounds_simplified.csv` (109MB): A simplified version of the foods and compounds dataset.
  - `desktop.ini`: System file.

- **compounds_presence/**: Contains binary (0/1) matrices of compound presence in foods and related analysis files.
  - `foodname_compound_presence_0_1.csv`: Binary presence matrix by food name (113MB).
  - `foodid_compound_presence_0_1.csv`: Binary presence matrix by food ID (113MB).
  - `food_compound_presence_0_1.csv`: A general binary presence matrix (113MB).
  - `foodname_compound_presence_0_1_filtered.csv`: Filtered binary presence matrix by food name (80MB).
  - `average_linkage_clusters.txt`: Text file related to average linkage clustering results.
  - `average_linkage_clusters_distance.txt`: Text file detailing distances for average linkage clusters.

### Clustering and Visualization Directories

- **clusters/**: Contains generated clusters of foods based on compound similarities.
  - `flavordb_clusters/`: FlavorDB clustering results.
  - `c_20/`, `c_30/`, `c_40/`: Subdirectories possibly containing clustering results with 20, 30, and 40 clusters, respectively.
  - `filtered_reorganized_clusters.txt`: Text file with filtered and reorganized cluster information.
  - `processed_sub_clusters.txt`: Processed sub-cluster data.
  - Various PNG files visualizing clusters (e.g., `hierarchical_clustering_dendrogram_complete_lastp_p10.png`, `kmeans_clusters.png`).
  - Contains various clustering results organized by method and parameters.

- **cluster_comparisons/**: Contains comparison results between different clustering approaches.
  - `original_foods/`: Clustering results for original food list.
  - `filtered_foods_union/`: Clustering results for filtered food list.
  
  - **clustering_comparison_results/**: Contains comparison results between different clustering approaches of embeddings and flavordb.

- **cluster_results/**: Contains detailed results of cluster analyses.
  - `cluster_comparison_results.txt`: General cluster comparison results.
  - `cluster_comparison_results_fullycoherent.txt`: Results focusing on fully coherent clusters.
  - Dated result files (e.g., `cluster_comparison_results_fullycoherent_19_11_2024_correct.txt`).
  - Contains comparison results between different clustering methods and cluster evaluation metrics.

- **correspondence_files/**: Contains cluster correspondence information.
  - CSV files mapping between different clustering methods and parameters (e.g., `correspondence_hierarchical_Hierarchical_ward__n_10_rank2_50.csv`, `correspondence_kmeans_K-means++_n_10_rank2_50.csv`).
  - CSV files with cluster comparison metrics (e.g., `cluster_comparison_metrics_expanded_refined_food_clusters.csv`).
  - Files aid in evaluating cluster correspondences and similarities.

- **visualization/**: Contains visualization outputs.
  - Dendrograms from hierarchical clustering (e.g., `hierarchical_clustering_dendrogram_average_lastp_p40.png`).
  - PCA and UMAP projections (e.g., `pca_projections_clusters_average_10_20.png`, `umap_projections_clusters_average_10_20.png`).
  - Cluster distribution visualizations (e.g., `food_conpounds_distribution.png`, `5clusters.png`).

- **old_plots/**: Contains older visualization outputs for reference.
  - Various dendrogram visualizations with different linkage methods (e.g., `hierarchical_clustering_dendrogram_complete.png`).
  - Silhouette score visualizations (e.g., `silhouette_score_clusters.png`).

### Supporting Directories

- **food_data/**: Contains food-related datasets.
  - `similar_foods.csv`: Lists of similar foods.
  - `common_and_unique_foods_comparison_flavrodb_clusteres580foods.csv`: Comparison of common and unique foods related to FlavorDB clusters.
  - `unique_foods_comparison_flavrodb_clusteres580foods.csv`: Data on unique foods related to FlavorDB clusters.

- **food_matches/**: Contains data on matching foods between datasets.
  - `food_matches_complete.csv`: Comprehensive food matches.
  - `food_matches_complete_modified.csv`: Modified version of the food matches.
  - `food_matches_complete_19_11_2024.csv`: Dated version of food matches.
  - These files likely map foods between FooDB and FlavorDB and provide match metrics.

- **documents/**: Contains documentation and analysis reports.
  - `sihouette_scores.docx`: Analysis of silhouette scores.
  - `correspondence_and_metrics_explanation.docx`: Document explaining correspondence and metrics.
  - `nexts steps.docx`: Document outlining next steps for the project.
  - Other documentation.

## Data Processing Pipeline

1. **Data Preparation**:
   - Foods and compounds are extracted from FooDB.
   - Compounds are categorized by taste (bitter, sweet, umami, other).
   - Binary presence matrices are created for compound-food relationships.

2. **Embedding Generation**:
   - Word2Vec and FastText models are trained for each taste category or all categories combined.
   - Embeddings are generated for each compound.
   - Compound embeddings are aggregated (e.g., averaging, concatenation if applicable by scripts like `generate_unified_embeddings.py`) to create unified food embeddings.
   - Multiple embedding sizes (e.g., 50-250 dimensions) and window sizes (e.g., 2-10) are tested.

3. **Clustering and Analysis**:
   - Foods are clustered using various algorithms (K-means, DBSCAN, Hierarchical).
   - Different linkage methods are evaluated for hierarchical clustering.
   - Clusters are compared with reference classifications (e.g., FlavorDB).
   - Visualizations are created using PCA and UMAP.

4. **Evaluation**:
   - Clustering results are evaluated using silhouette scores and other metrics.
   - Precision, recall, and other metrics are calculated for cluster comparisons and correspondence.
   - Food matches between different databases are analyzed.

## Key Technologies

- **Python**: Core programming language
- **Pandas**: Data manipulation and processing
- **Word2Vec (Gensim)** & **FastText (Gensim)**: Embedding generation for compounds
- **Scikit-learn**: Clustering algorithms (K-means, DBSCAN) and evaluation metrics
- **UMAP** and **PCA**: Dimensionality reduction for visualization
- **Matplotlib**: Visualization of clusters and results
- **PyTorch**: Experimental autoencoder implementations

## Research Focus

This project appears to be part of a thesis on food compound analysis, focusing on:
- Understanding relationships between compounds and taste categories
- Exploring how food compounds can be represented in vector space
- Clustering foods based on chemical similarity rather than traditional categories
- Comparing embedding-based clustering with traditional food categorizations
- Evaluating different embedding and clustering approaches

## Note on Large Files

Many files in this repository are quite large:
- Embedding files (*.pkl) can range from 1.5MB to over 7MB for unified embeddings.
- Word2Vec/FastText models (*.model) are typically 2.9MB to 49MB or larger, with associated `.npy` files reaching hundreds of MBs (e.g., 763MB for some FastText ngrams).
- FooDB database files and derived CSVs can be up to 487MB or more.
- Ordered compounds files are around 130MB to 200MB.

These large files are excluded from Git using the `.gitignore` file to comply with GitHub size limitations. 