# Food Compound Analysis and Clustering

This repository contains code and data for analyzing food compounds, generating embeddings, and clustering foods based on their compound profiles. The project focuses on understanding the relationship between food compounds and taste categories (bitter, sweet, umami, and other).

## Project Structure

### Core Directories

- **scripts/**: Contains Python scripts for data processing, embedding generation, and clustering
  - `word_embeddings.py`: Generates Word2Vec embeddings for compounds across different taste categories
  - `cluster_embeddings.py`: Applies clustering algorithms (K-means, DBSCAN) to food embeddings
  - `visualize_clusters.py`: Creates visualizations of clusters using PCA and UMAP
  - `compare_clusters_embedding_presence.py`: Compares different clustering approaches
  - `check_foods_flavordb_foodb.py`: Compares foods between FlavorDB and FooDB databases
  - `thesis_preprocess_foodb.py`: Preprocesses the FooDB data for the thesis
  - `foods_compounds_0_1.py`: Creates binary presence matrices for compounds in foods
  - `generate_unified_embeddings.py`: Combines embeddings from different taste categories
  - `autoencoder_try.py` & `pytorch_try.py`: Experimental scripts for autoencoders
  - Other utility scripts for data processing and analysis

- **compounds/**: Contains compound data categorized by taste
  - `compounds_bitter.csv`: Bitter compounds data (3.2MB)
  - `compounds_sweet.csv`: Sweet compounds data (3.2MB)
  - `compounds_umami.csv`: Umami compounds data (3.2MB)
  - `compounds_other.csv`: Other compounds data (3.2MB)

- **ordered_compounds/**: Contains ordered compounds data per food for each taste category
  - `ordered_compounds_per_food_bitter.csv`: Ordered bitter compounds per food (131MB)
  - `ordered_compounds_per_food_sweet.csv`: Ordered sweet compounds per food (131MB)
  - `ordered_compounds_per_food_umami.csv`: Ordered umami compounds per food (131MB)
  - `ordered_compounds_per_food_other.csv`: Ordered other compounds per food (131MB)

### Data and Results Directories

- **embeddings/**: Contains generated embeddings for foods and compounds
  - Multiple embedding files with different dimensions (50, 100, 150, 200, 250) and window sizes
  - Files organized by taste category (bitter, sweet, umami, other)
  - Contains both raw and filtered embeddings

- **embeddings_data/**: Contains processed embedding data and latent representations
  - Various unified embeddings in pickle format
  - Latent dataframes from dimensionality reduction

- **models/**: Contains trained Word2Vec models
  - Models for different tastes (bitter, sweet, umami, other)
  - Models with different embedding dimensions and parameters

- **foodb/**: Contains data from the FooDB database
  - `FoodDB.csv`: Main FooDB database (65MB)
  - `food_compound_foodb.csv`: Food-compound relationships (195MB)
  - `food_compound_foodb_grouped.csv`: Grouped food-compound data (487MB)
  - Files separated by taste categories (bitter, sweet, umami, other)

- **compounds_presence/**: Contains binary (0/1) matrices of compound presence in foods
  - `foodname_compound_presence_0_1.csv`: Binary presence matrix by food name (113MB)
  - `foodid_compound_presence_0_1.csv`: Binary presence matrix by food ID (113MB)

### Clustering and Visualization Directories

- **clusters/**: Contains generated clusters of foods based on compound similarities
  - Various clustering results organized by method
  - `flavordb_clusters/`: FlavorDB clustering results
  - Processed cluster assignments

- **cluster_data/**: Contains cluster data files and analysis results
  - Clustering results in CSV format
  - Comparison data between different clustering approaches

- **cluster_comparisons/**: Contains comparison results between different clustering approaches
  - `original_foods/`: Clustering results for original food list
  - `filtered_foods_union/`: Clustering results for filtered food list

- **cluster_results/**: Contains detailed results of cluster analyses
  - Comparison results between different clustering methods
  - Cluster evaluation metrics

- **correspondence_files/**: Contains cluster correspondence information
  - Files mapping between different clustering methods
  - Metrics for evaluating cluster correspondences

- **visualization/**: Contains visualization outputs
  - Dendrograms from hierarchical clustering
  - PCA and UMAP projections
  - Cluster distribution visualizations

- **old_plots/**: Contains older visualization outputs for reference
  - Various dendrogram visualizations with different linkage methods
  - Silhouette score visualizations

### Supporting Directories

- **food_data/**: Contains food-related datasets
  - Similar foods comparisons
  - Unique foods lists

- **food_comparison_data/**: Contains data comparing foods across different datasets
  - Comparison between different food databases
  - Food similarity metrics

- **food_matches/**: Contains data on matching foods between datasets
  - Mappings between FooDB and FlavorDB
  - Food match metrics

- **text_files/**: Contains miscellaneous text data
  - Global ordering information
  - Missing foods records

- **documents/**: Contains documentation and analysis reports
  - Silhouette scores analysis
  - Other documentation

## Data Processing Pipeline

1. **Data Preparation**:
   - Foods and compounds are extracted from FooDB
   - Compounds are categorized by taste (bitter, sweet, umami, other)
   - Binary presence matrices are created for compound-food relationships

2. **Embedding Generation**:
   - Word2Vec models are trained for each taste category
   - Embeddings are generated for each compound
   - Compound embeddings are aggregated to create unified food embeddings
   - Multiple embedding sizes (50-250 dimensions) and window sizes (2-10) are tested

3. **Clustering and Analysis**:
   - Foods are clustered using various algorithms (K-means, DBSCAN, Hierarchical)
   - Different linkage methods are evaluated for hierarchical clustering
   - Clusters are compared with reference classifications (FlavorDB)
   - Visualizations are created using PCA and UMAP

4. **Evaluation**:
   - Clustering results are evaluated using silhouette scores
   - Precision and recall metrics are calculated for cluster comparisons
   - Food matches between different databases are analyzed

## Key Technologies

- **Python**: Core programming language
- **Pandas**: Data manipulation and processing
- **Word2Vec (Gensim)**: Embedding generation for compounds
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
- Embedding files (*.pkl) can range from 1.5MB to several GB
- Word2Vec models (*.model) are typically 26MB or larger
- FooDB database files can be up to 487MB

These large files are excluded from Git using the `.gitignore` file to comply with GitHub size limitations. 