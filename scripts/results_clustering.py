import pandas as pd

# Load and examine the results
results_df = pd.read_csv("/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/clustering_comparison_results/clustering_algorithm_comparison_all_embeddings.csv")

# Show top performing configurations
print("\nTop performing configurations:")
print(results_df.sort_values(['ari_score', 'clustered_percentage'], ascending=[False, False]).head(10))

# Show summary by algorithm type
print("\nSummary by algorithm type:")
print(results_df.groupby('algorithm')['ari_score'].agg(['mean', 'max', 'min']).sort_values('max', ascending=False))