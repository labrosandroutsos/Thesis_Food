import pandas as pd 
import numpy as np

reorganized_clusters_path = 'filtered_reorganized_clusters.txt'

# Load the presence-based clusters from the reorganized_clusters.txt file
presence_clusters_dict = {}

# with open(reorganized_clusters_path, 'r') as file:
#     current_cluster = None
#     for line in file:
#         line = line.strip()
#         if not line:  # Skip empty lines
#             continue
        
#         if line.startswith('Cluster'):
#             current_cluster = line.split('Cluster ')[-1].strip(':')
#             presence_clusters_dict[current_cluster] = []
#         elif current_cluster is not None:  # Add food name to the current cluster
#             presence_clusters_dict[current_cluster].append(line)

with open(reorganized_clusters_path, 'r', encoding='utf-8') as file:
    current_cluster = None
    for line in file:
        if "Cluster" in line:
            current_cluster = int(line.strip().split(" ")[1].replace(":", ""))
            presence_clusters_dict[current_cluster] = []
        elif line.strip():  # Add food name to the current cluster
            presence_clusters_dict[current_cluster].append(line.strip())

# Flatten the dictionary to get all food names
all_foods_reorganized = {food for foods in presence_clusters_dict.values() for food in foods}
print(all_foods_reorganized)
print(len(all_foods_reorganized))

df_foodb = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_presence/foodname_compound_presence_0_1.csv', sep=';', index_col=0)

print(df_foodb)

final_df = df_foodb.rename(index={'Rosé wine': 'Rose wine'})

filtered_df = final_df[final_df.index.isin(all_foods_reorganized)]

print(filtered_df)

filtered_df.to_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_presence/foodname_compound_presence_0_1_filtered.csv', sep=';', index=True)
# embedding_sizes = [50, 100, 150, 200, 250]

# # Load the final unified embeddings and filter foods
# for size in embedding_sizes:
#     final_df = pd.read_pickle(f'C:/Users/labro/Downloads/Thesis_Food/embeddings/final_unified_embeddings_aggregated_{size}.pkl')
    
#     # Replace 'Rosé wine' with 'Rose wine'
#     final_df['food_name'] = final_df['food_name'].replace('Rosé wine', 'Rose wine')
    
#     # Find missing foods
#     missing_foods = all_foods_reorganized - set(final_df['food_name'])
#     if missing_foods:
#         print(f"\nFoods missing in {size} embeddings:")
#         for food in missing_foods:
#             print(f"- {food}")
    
#     # Keep only foods that are in all_foods_reorganized
#     filtered_df = final_df[final_df['food_name'].isin(all_foods_reorganized)]
    
#     # Save the filtered dataframe
#     filtered_df.to_pickle(f'C:/Users/labro/Downloads/Thesis_Food/embeddings/filtered_unified_embeddings_aggregated_{size}.pkl')
#     print(f"\nSaved filtered embeddings for size {size}. Shape: {filtered_df.shape}")

# X_flat = np.array(final_df['unified_embedding'].tolist())