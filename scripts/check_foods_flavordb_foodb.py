import pandas as pd

reorganized_clusters_path = 'archive_clusters/revised_reorganized_clusters.txt'
# reorganized_clusters_path = 'compounds_presence/average_linkage_clusters.txt'
# flavordb_clusters_path = '../clusters/flavordb_clusters/processed_flavordb_clusters.txt'
flavordb_clusters_path = 'clusters/flavordb_clusters/flavordb_clusters.txt'
# flavordb_clusters_path = 'clusters/flavordb_clusters/filtered_flavordb_clusters_683.txt'
# flavordb_clusters_path = 'compounds_presence/average_linkage_clusters.txt'
# flavordb_clusters_path = 'misc_clusters/expanded_refined_food_clusters_683_CHATGPT_CLUSTERS.txt'
import re
import pandas as pd

def read_cluster_file(file_path, is_reorganized=False):
    clusters = {}
    current_cluster = None
    
    def split_foods(line):
        # Use a regular expression to split on commas that are not within parentheses
        return re.split(r',\s*(?![^()]*\))', line)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            if line.startswith('Cluster'):
                current_cluster = line.split('Cluster ')[-1].strip(':')
                clusters[current_cluster] = []
            elif line and current_cluster is not None:
                if is_reorganized:
                    # For reorganized clusters, each food is on a new line
                    clusters[current_cluster].append(line)
                else:
                    # For FlavorDB clusters, foods are comma-separated
                    foods = [food.strip() for food in split_foods(line)]
                    clusters[current_cluster].extend(foods)

    return clusters

# # Read both cluster files
reorganized_clusters = read_cluster_file(reorganized_clusters_path, is_reorganized=True)
# flavordb_clusters = read_cluster_file(flavordb_clusters_path, is_reorganized=False)
flavordb_clusters = read_cluster_file(flavordb_clusters_path, is_reorganized=False)

# Get all unique foods from both sources
reorganized_foods = set()
for foods in reorganized_clusters.values():
    reorganized_foods.update(foods)

flavordb_foods = set()
for foods in flavordb_clusters.values():
    flavordb_foods.update(foods)

# Convert sets to lists and pad the shorter list with None
reorganized_foods_list = list(reorganized_foods)
flavordb_foods_list = list(flavordb_foods)

print(len(reorganized_foods_list))
print(len(flavordb_foods_list))
# print(flavordb_clusters)

max_length = max(len(reorganized_foods_list), len(flavordb_foods_list))
reorganized_foods_list.extend([None] * (max_length - len(reorganized_foods_list)))
flavordb_foods_list.extend([None] * (max_length - len(flavordb_foods_list)))

# Create a DataFrame with two columns for unique foods
unique_foods_df = pd.DataFrame({
    'Reorganized_from_chatgpt_Foods': reorganized_foods_list,
    'Average_linkage_683_Foods': flavordb_foods_list
})

# Save the DataFrame to a CSV file
# output_csv = 'unique_foods_comparison_flavrodb_clusteres580foods.csv'
# unique_foods_df.to_csv(output_csv, index=False)
# print(f"Unique foods saved to {output_csv}")

# # Find mismatches
only_in_reorganized = reorganized_foods - flavordb_foods
only_in_flavordb = flavordb_foods - reorganized_foods
common_foods = reorganized_foods.intersection(flavordb_foods)

print(len(common_foods))
print(len(only_in_reorganized))
print(len(only_in_flavordb))
# print((only_in_flavordb))


# First, create a DataFrame with matching foods
common_foods = list(reorganized_foods.intersection(flavordb_foods))
matching_df = pd.DataFrame({
    'Reorganized_from_chatgpt_Foods': common_foods,
    'Average_linkage_683_Foods': common_foods
})

# Get unique foods from each source
only_reorganized = list(reorganized_foods - flavordb_foods)
only_flavordb = list(flavordb_foods - reorganized_foods)

# Create DataFrames for unique foods
max_unique_length = max(len(only_reorganized), len(only_flavordb))
only_reorganized.extend([None] * (max_unique_length - len(only_reorganized)))
only_flavordb.extend([None] * (max_unique_length - len(only_flavordb)))

unique_df = pd.DataFrame({
    'Reorganized_from_chatgpt_Foods': only_reorganized,
    'Average_linkage_683_Foods': only_flavordb
})

# Concatenate the matching and unique DataFrames
unique_foods_df = pd.concat([matching_df, unique_df], ignore_index=True)

# Save the DataFrame to a CSV file
# output_csv = '../food_data/common_and_unique_foods_comparison_flavrodb_clusteres580foods.csv'
# unique_foods_df.to_csv(output_csv, index=False)
# print(f"Unique foods saved to {output_csv}")

# Filter clusters to keep only common foods
filtered_reorganized_clusters = {
    cluster: [food for food in foods if food in common_foods]
    for cluster, foods in reorganized_clusters.items()
}

filtered_flavordb_clusters = {
    cluster: [food for food in foods if food in common_foods]
    for cluster, foods in flavordb_clusters.items()
}

# # Save the filtered clusters to new text files
# reorganized_output_file = 'filtered_reorganized_clusters.txt'
# with open(reorganized_output_file, 'w', encoding='utf-8') as f:
#     for cluster_name, foods in filtered_reorganized_clusters.items():
#         f.write(f"Cluster {cluster_name}:\n")
#         f.write("\n".join(foods) + "\n\n")

# flavordb_output_file = 'filtered_flavordb_clusters.txt'
# with open(flavordb_output_file, 'w', encoding='utf-8') as f:
#     for cluster_name, foods in filtered_flavordb_clusters.items():
#         f.write(f"Cluster {cluster_name}:\n")
#         f.write(", ".join(foods) + "\n\n")

# print(f"Filtered reorganized clusters saved to {reorganized_output_file}")
# print(f"Filtered FlavorDB clusters saved to {flavordb_output_file}")


# save also the full clusters to new text files. I want filtered clusters items + the foods that are unique for each cluster
# # reorganized_output_file = 'filtered_reorganized_clusters.txt'
# reorganized_output_file = 'filtered_reorganized_clusters.txt'
# with open(reorganized_output_file, 'w', encoding='utf-8') as f: 
#     for cluster_name, foods in reorganized_clusters.items():
#         f.write(f"Cluster {cluster_name}:\n")
#         f.write("\n".join(foods) + "\n\n")

# # flavordb_output_file = 'filtered_flavordb_clusters.txt'
# flavordb_output_file = 'filtered_flavordb_clusters.txt'
# with open(flavordb_output_file, 'w', encoding='utf-8') as f:
#     for cluster_name, foods in flavordb_clusters.items():
#         f.write(f"Cluster {cluster_name}:\n")
#         f.write(", ".join(foods) + "\n\n")   

# print(f"Full reorganized clusters saved to {reorganized_output_file}")

# # Print statistics
# print("Statistics:")
# print(f"Total foods in reorganized clusters: {len(reorganized_foods)}")
# print(f"Total foods in FlavorDB clusters: {len(flavordb_foods)}")
# print(f"Foods in both: {len(common_foods)}")
# print(f"Foods only in reorganized: {len(only_in_reorganized)}")
# print(f"Foods only in FlavorDB: {len(only_in_flavordb)}")

# # Print detailed mismatches
# print("\nFoods only in reorganized clusters:")
# for food in sorted(only_in_reorganized):
#     print(f"- {food}")

# print("\nFoods only in FlavorDB:")
# for food in sorted(only_in_flavordb):
#     print(f"- {food}")

# # Check cluster assignments for common foods
# print("\nDifferent cluster assignments for common foods:")
# for food in common_foods:
#     reorg_cluster = next(cluster for cluster, foods in reorganized_clusters.items() if food in foods)
#     flavordb_cluster = next(cluster for cluster, foods in flavordb_clusters.items() if food in foods)
    
#     # if reorg_cluster != flavordb_cluster:
#     #     print(f"Food: {food}")
#     #     print(f"  Reorganized cluster: {reorg_cluster}")
#     #     print(f"  FlavorDB cluster: {flavordb_cluster}")

# # # Save results to file
# output_file = 'cluster_comparison_results_fullycoherent_20_11_2024_correct.txt'
# with open(output_file, 'w', encoding='utf-8') as f:
#     f.write("Cluster Comparison Results\n")
#     f.write("=========================\n\n")
    
#     f.write("Statistics:\n")
#     f.write(f"Total foods in reorganized clusters: {len(reorganized_foods)}\n")
#     f.write(f"Total foods in FlavorDB clusters: {len(flavordb_foods)}\n")
#     f.write(f"Foods in both: {len(common_foods)}\n")
#     f.write(f"Foods only in reorganized: {len(only_in_reorganized)}\n")
#     f.write(f"Foods only in FlavorDB: {len(only_in_flavordb)}\n\n")
    
#     f.write("Foods only in reorganized clusters:\n")
#     for food in sorted(only_in_reorganized):
#         f.write(f"- {food}\n")
    
#     f.write("\nFoods only in FlavorDB:\n")
#     for food in sorted(only_in_flavordb):
#         f.write(f"- {food}\n")
    
#     # f.write("\nDifferent cluster assignments for common foods:\n")
#     # for food in common_foods:
#     #     reorg_cluster = next(cluster for cluster, foods in reorganized_clusters.items() if food in foods)
#     #     flavordb_cluster = next(cluster for cluster, foods in flavordb_clusters.items() if food in foods)
        
#     #     if reorg_cluster != flavordb_cluster:
#     #         f.write(f"Food: {food}\n")
#     #         f.write(f"  Reorganized cluster: {reorg_cluster}\n")
#     #         f.write(f"  FlavorDB cluster: {flavordb_cluster}\n")

# print(f"\nResults saved to {output_file}")


# # final_df = pd.read_pickle(f'C:/Users/labro/Downloads/Thesis_Food/embeddings/final_unified_embeddings_aggregated_50.pkl')
# # print(final_df)
# # print(final_df.shape)


# import pandas as pd
# from difflib import get_close_matches

# # Load the comparison file
# comparison_file_path = 'cluster_comparison_results_fullycoherent_19_11_2024_correct.txt'

# # # Initialize lists to capture foods unique to each representation and shared items
# unique_to_reorganized = []
# unique_to_flavordb = []

# # # Process the file to separate foods into these lists based on the section they're found in
# with open(comparison_file_path, 'r') as file:
#     current_section = None
#     for line in file:
#         line = line.strip()
#         if "Foods unique to Reorganized Clusters:" in line:
#             current_section = "reorganized"
#         elif "Foods unique to FlavorDB Clusters:" in line:
#             current_section = "flavordb"
#         elif line and current_section == "reorganized":
#             unique_to_reorganized.append(line)
#         elif line and current_section == "flavordb":
#             unique_to_flavordb.append(line)

# # print(unique_to_reorganized)
# # print(unique_to_flavordb)
# # Perform matching for all foods with a lower cutoff to catch more potential matches
# similar_foods = []
# cutoff = 0.6  # Lower cutoff to catch more potential matches

# # Match foods from reorganized to flavordb
# for food in only_in_reorganized:
#     close_matches = get_close_matches(food, list(only_in_flavordb), n=5, cutoff=cutoff)
#     if close_matches:
#         for match in close_matches:
#             similarity_score = sum(a == b for a, b in zip(food.lower(), match.lower())) / max(len(food), len(match))
#             similar_foods.append({
#                 'Reorganized_Food': food,
#                 'FlavorDB_Food': match,
#                 'Similarity_Score': round(similarity_score, 3),
#                 'Source': 'Reorganized'
#             })
#     # else:
#     #     # Include unmatched foods with empty matches
#     #     similar_foods.append({
#     #         'Reorganized_Food': food,
#     #         'FlavorDB_Food': '',
#     #         'Similarity_Score': 0.0,
#     #         'Source': 'Reorganized'
#     #     })

# # Match foods from flavordb to reorganized (to catch any remaining unmatched foods)
# for food in only_in_flavordb:
#     if not any(entry['FlavorDB_Food'] == food for entry in similar_foods):
#         close_matches = get_close_matches(food, list(only_in_reorganized), n=5, cutoff=cutoff)
#         if close_matches:
#             for match in close_matches:
#                 similarity_score = sum(a == b for a, b in zip(food.lower(), match.lower())) / max(len(food), len(match))
#                 similar_foods.append({
#                     'Reorganized_Food': match,
#                     'FlavorDB_Food': food,
#                     'Similarity_Score': round(similarity_score, 3),
#                     'Source': 'FlavorDB'
#                 })
#         # else:
#         #     # Include unmatched foods with empty matches
#         #     similar_foods.append({
#         #         'Reorganized_Food': '',
#         #         'FlavorDB_Food': food,
#         #         'Similarity_Score': 0.0,
#         #         'Source': 'FlavorDB'
#             # })

# # Create DataFrame from similar foods
# similar_foods_df = pd.DataFrame(similar_foods)

# # Sort by similarity score in descending order
# similar_foods_df = similar_foods_df.sort_values('Similarity_Score', ascending=False)

# # Display the DataFrame
# print("\nSimilar Foods DataFrame:")
# print(similar_foods_df)

# # Save DataFrame to CSV
# output_csv = 'food_matches_complete_19_11_2024.csv'
# similar_foods_df.to_csv(output_csv, index=False)
# print(f"\nAll food matches saved to {output_csv}")

# # Print some statistics
# print("\nMatching Statistics:")
# print(f"Total pairs found: {len(similar_foods_df)}")
# print(f"Matches with similarity score > 0.8: {len(similar_foods_df[similar_foods_df['Similarity_Score'] > 0.8])}")
# print(f"Matches with similarity score 0.6-0.8: {len(similar_foods_df[(similar_foods_df['Similarity_Score'] >= 0.6) & (similar_foods_df['Similarity_Score'] <= 0.8)])}")
# print(f"Unmatched foods (similarity score 0): {len(similar_foods_df[similar_foods_df['Similarity_Score'] == 0])}")

# # Load the CSV file
# # output_csv = 'food_matches_complete.csv'
# # df = pd.read_csv(output_csv)

# # # Function to lowercase the second word in a string, handling NaN values
# # def lowercase_second_word(s):
# #     if pd.isna(s):
# #         return s  # Return NaN unchanged
# #     words = s.split()
# #     if len(words) > 1:
# #         words[1] = words[1].lower()
# #     return ' '.join(words)

# # # Apply the function to the FlavorDB_Food column
# # df['FlavorDB_Food'] = df['FlavorDB_Food'].apply(lowercase_second_word)

# # # Display the modified DataFrame
# # print(df)

# # # Optionally, save the modified DataFrame back to a CSV
# # df.to_csv('food_matches_complete_modified.csv', index=False)
# # print("\nModified CSV saved as 'food_matches_complete_modified.csv'")

# # output = 'food_matches_complete_modified.csv'
# # df = pd.read_csv(output)

# # # Get the unique values of the first two columns
# # first_column_unique_values = df.iloc[:, 0].unique()
# # second_column_unique_values = df.iloc[:, 1].unique()

# # # Calculate the total number of unique values
# # total_first_column = len(first_column_unique_values)
# # total_second_column = len(second_column_unique_values)

# # # Calculate differences
# # only_in_first = set(first_column_unique_values) - set(second_column_unique_values)
# # only_in_second = set(second_column_unique_values) - set(first_column_unique_values)
# # common_values = set(first_column_unique_values).intersection(set(second_column_unique_values))

# # # Display the unique values and statistics
# # print("Unique values in the first column:")
# # print(first_column_unique_values)
# # print(f"Total unique values in the first column: {total_first_column}")

# # print("\nUnique values in the second column:")
# # print(second_column_unique_values)
# # print(f"Total unique values in the second column: {total_second_column}")

# # print("\nCommon values in both columns:")
# # print(common_values)
# # print(f"Total common values: {len(common_values)}")

# # print("\nValues only in the first column:")
# # print(only_in_first)
# # print(f"Total values only in the first column: {len(only_in_first)}")

# # print("\nValues only in the second column:")
# # print(only_in_second)
# # print(f"Total values only in the second column: {len(only_in_second)}")

# # flavordb_clusters_path = 'flavordb_clusters.txt'
# # def read_and_process_flavordb_clusters(file_path):
# #     clusters = {}
# #     current_cluster = None

# #     def lowercase_second_word(s):
# #         words = s.split()
# #         if len(words) > 1:
# #             words[1] = words[1].lower()
# #         return ' '.join(words)

# #     with open(file_path, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             line = line.strip()
# #             if not line:  # Skip empty lines
# #                 continue

# #             if line.startswith('Cluster'):
# #                 current_cluster = line.split('Cluster ')[-1].strip(':')
# #                 clusters[current_cluster] = []
# #             elif line and current_cluster is not None:
# #                 # Split by comma, apply lowercase to second word, and strip whitespace
# #                 foods = [lowercase_second_word(food.strip()) for food in line.split(',')]
# #                 clusters[current_cluster].extend(foods)

# #     return clusters

# # # Read and process the FlavorDB clusters
# # flavordb_clusters = read_and_process_flavordb_clusters(flavordb_clusters_path)

# # # Display the processed clusters
# # for cluster_name, foods in flavordb_clusters.items():
# #     print(f"Cluster {cluster_name}:")
# #     print(", ".join(foods))


# # # Save the processed clusters to a new file
# # output_file = 'processed_flavordb_clusters.txt'
# # with open(output_file, 'w', encoding='utf-8') as f:
# #     for cluster_name, foods in flavordb_clusters.items():
# #         f.write(f"Cluster {cluster_name}:\n")
# #         f.write(", ".join(foods) + "\n\n")

# # print(f"Processed clusters saved to {output_file}")