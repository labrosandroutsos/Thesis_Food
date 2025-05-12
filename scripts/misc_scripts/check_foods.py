import pandas as pd

def check_foods():
    # Load the original 'unified_clusters_comparison.csv' to get the full list of 962 foods
    # unified_comparison_df = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/unified_clusters_comparison.csv')
    unified_comparison_df = pd.read_csv('/home/lamprosandroutsos/Documents/Thesis/Thesis_Food/unified_clusters_comparison.csv')
    original_food_names = set(unified_comparison_df['food_name'].unique())

    print(len(original_food_names))
    # Load the foods in the reorganized_clusters.txt to identify missing foods
    reorganized_clusters_path = 'C:/Users/labro/Downloads/reorganized_clusters.txt'
    reorganized_food_names = set()

    with open(reorganized_clusters_path, 'r') as file:
        current_cluster = None
        for line in file:
            if "Cluster" not in line and line.strip():  # Skip cluster headers and blank lines
                reorganized_food_names.add(line.strip())
    print(len(reorganized_food_names))
    # Identify missing foods by comparing the two sets
    missing_foods = original_food_names - reorganized_food_names
    print(len(missing_foods), list(missing_foods)[:10])  # Display the number and a sample of missing foods for review
    print(missing_foods)

# Define a function to save the refined clusters to a text file
def save_clusters_to_file(cluster_dict, file_name):
    with open(file_name, 'w') as file:
        for cluster, foods in cluster_dict.items():
            file.write(f"Cluster {cluster}:\n")
            for food in foods:
                file.write(f"{food}\n")
            file.write("\n")

# Let's re-categorize each food item comprehensively to ensure that all items are fully accounted for in the updated clusters.

cluster_file_path = 'C:/Users/labro/Downloads/reorganized_clusters.txt'
presence_clusters_dict = {}

# # Load the clusters into a dictionary for structured analysis
# with open(cluster_file_path, 'r') as file:
#     current_cluster = None
#     for line in file:
#         if "Cluster" in line:
#             current_cluster = int(line.strip().split(" ")[1].replace(":", ""))
#             presence_clusters_dict[current_cluster] = []
#         elif line.strip():  # Add food name to the current cluster
#             presence_clusters_dict[current_cluster].append(line.strip())


# presence_clusters_dict[2] = sorted([
#     'Red grape', 'Water spinach', 'Tortilla', 'Cantaloupe melon', 'Tofu', 'Potato bread', 'Black raisin', 
#     'Pea shoots', 'White mulberry', 'Miso', 'Plantain', 'Piki bread', 'Mundu', 'White bread', 'Bagel', 
#     'Taco shell', 'Lantern fruit', 'Raisin bread', 'Soy milk', 'Cubanelle pepper', 'Hawthorn', 'Chineese plum', 
#     'Curry powder', 'Gentiana lutea', 'Crosne', 'Rabbiteye blueberry', 'Green lentil', 'Soy sauce', 
#     'Tostada shell', 'Red clover', 'Goji', 'Cape gooseberry', 'Flour', 'Green cabbage', 'Clementine', 
#     'Oat bread', 'Hibiscus tea', 'Yau choy', 'Sour orange', 'Juniperus communis', 'Green apple', 'Jalapeno pepper', 
#     'Eddoe', 'Mikan', 'Zwieback', 'Heart of palm', 'Arabica coffee', 'Yali pear', 'Green plum', 'Soy cream', 
#     'Monk fruit', 'Guarana', 'Green grape', 'Castanospermum australe', 'Mate', 'Morchella (Morel)', 
#     'Partridge berry', 'Blackberry', 'Wonton wrapper', 'Rye bread', 'Cannellini bean', 'Pitaya', 
#     'Robusta coffee', 'Pita bread', 'Black plum', 'Rice bread', 'Albizia gummifera', 'Cornbread', 'Bulgur', 
#     'Wheat bread', 'Semolina', 'Wampee', 'Iceberg lettuce', 'Acorn squash'
# ])
# # Save the fully updated clusters to a new text file
# output_file_path = 'C:/Users/labro/Downloads/final_fully_coherent_clusters.txt'
# save_clusters_to_file(presence_clusters_dict, output_file_path)


check_foods()