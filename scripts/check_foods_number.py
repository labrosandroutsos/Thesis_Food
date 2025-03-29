import pandas as pd

# Load the comparison CSV with 962 foods
comparison_df = pd.read_csv('../comparison_13_clusters.csv')
unique_foods_in_comparison = set(comparison_df['food_name'].unique())

# Load foods from reorganized_clusters.txt
# reorganized_clusters_path = 'C:/Users/labro/Downloads/reorganized_clusters.txt'
clusters_path = 'C:/Users/labro/Downloads/average_linkage_clusters_reclusteredwithKnowledge_24_11.txt'

presence_clusters_dict = {}
current_cluster = None

with open(clusters_path, 'r') as file:
    for line in file:
        line = line.strip()
        print(line)
        if not line:  # Skip empty lines
            continue
        
        if line.startswith('Cluster'):
            current_cluster = line.split('Cluster ')[-1].strip(':')
            presence_clusters_dict[current_cluster] = []
        elif current_cluster is not None:  # Add food name to the current cluster
            presence_clusters_dict[current_cluster].append(line)

# Flatten the dictionary to get all food names
all_foods_reorganized = {food for foods in presence_clusters_dict.values() for food in foods}

# # Identify missing foods
# missing_foods = unique_foods_in_comparison - all_foods_reorganized
# print(f"Total foods missing: {len(missing_foods)}")
# print("Example missing foods:", list(missing_foods)[:10])

# Additional statistics
print(f"\nTotal foods in comparison CSV: {len(unique_foods_in_comparison)}")
print(f"Total foods in reorganized clusters: {len(all_foods_reorganized)}")

# # Save missing foods to a file
# with open('missing_foods.txt', 'w') as f:
#     f.write("Missing foods:\n")
#     for food in sorted(missing_foods):
#         f.write(f"- {food}\n")
# print("\nMissing foods have been saved to 'missing_foods.txt'")