import pandas as pd
import json
import os

# Read the JSON file
with open('../categories.json', 'r') as f:
    categories_data = json.load(f)

# Convert to DataFrame
df_categories = pd.DataFrame(categories_data)
print(f"Loaded categories file with {len(df_categories)} rows")

# Create a dictionary to store clusters
clusters = {}

# Extract foods from entities and create clusters
for idx, row in df_categories.iterrows():
    entities = row['entities']  # Should already be a list of dictionaries
    for entity in entities:
        category = entity['category_readable']
        food_name = entity['entity_alias_readable']
        
        if category not in clusters:
            clusters[category] = []
        
        if food_name:  # Only add if food name is not empty
            clusters[category].append(food_name)

# Remove duplicates from each cluster
for category in clusters:
    clusters[category] = list(set(clusters[category]))

# Save clusters to a file
output_txt = 'flavordb_clusters.txt'
with open(output_txt, 'w', encoding='utf-8') as f:
    for cluster_name, foods in sorted(clusters.items()):
        f.write(f"Cluster {cluster_name}:\n")
        f.write(", ".join(sorted(foods)))
        f.write("\n\n")
print(f"\nSaved {output_txt}")
print(f"File size: {os.path.getsize(output_txt)} bytes")

# Create a DataFrame with food names and their cluster labels
food_clusters = []
for cluster_name, foods in clusters.items():
    for food in sorted(foods):
        food_clusters.append({
            'food_name': food,
            'cluster': cluster_name,
        })

df_clusters = pd.DataFrame(food_clusters)

# Save the DataFrame
output_clusters = 'flavordb_food_clusters.csv'
df_clusters.to_csv(output_clusters, index=False)
print(f"\nSaved {output_clusters}")
print(f"File size: {os.path.getsize(output_clusters)} bytes")

# Print statistics and samples
print(f"\nNumber of clusters: {len(clusters)}")
for cluster_name, foods in sorted(clusters.items()):
    print(f"\n{cluster_name}: {len(foods)} foods")
    print("Sample foods:", ", ".join(sorted(foods)[:5]))