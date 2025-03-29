import pandas as pd

# # Load the original reorganized clusters to append the missing foods to appropriate clusters
# reorganized_clusters_path = 'C:/Users/labro/Downloads/reorganized_clusters.txt'
# presence_clusters_dict = {}

# # Rebuild the dictionary from reorganized clusters file
# with open(reorganized_clusters_path, 'r') as file:
#     current_cluster = None
#     for line in file:
#         if "Cluster" in line:
#             current_cluster = int(line.strip().split(" ")[1].replace(":", ""))
#             presence_clusters_dict[current_cluster] = []
#         elif line.strip():  # Add food name to the current cluster
#             presence_clusters_dict[current_cluster].append(line.strip())

# # List of missing foods to be added back to appropriate clusters
# missing_foods = {
#     'White bread', 'Lantern fruit', 'Cannellini bean', 'Tofu', 'Bagel', 'Castanospermum australe', 
#     'Taco shell', 'Rye bread', 'Heart of palm', 'Raisin bread', 'Hibiscus tea', 'Gentiana lutea', 
#     'Morchella (Morel)', 'Curry powder', 'Soy cream', 'Green lentil', 'Plantain', 'Eddoe', 
#     'Rabbiteye blueberry', 'Red clover', 'Cornbread', 'Pitaya', 'Zwieback', 'Yau choy', 
#     'White mulberry', 'Green apple', 'Mikan', 'Jalapeno pepper', 'Guarana', 'Pea shoots', 
#     'Cantaloupe melon', 'Piki bread', 'Cubanelle pepper', 'Iceberg lettuce', 'Black plum', 
#     'Rice bread', 'Soy sauce', 'Wampee', 'Wonton wrapper', 'Crosne', 'Arabica coffee', 
#     'Oat bread', 'Green grape', 'Tostada shell', 'Robusta coffee', 'Clementine', 
#     'Juniperus communis', 'Mate', 'Green plum', 'Partridge berry', 'Soy milk', 'Water spinach', 
#     'Miso', 'Yali pear', 'Tortilla', 'Mundu', 'Blackberry', 'Flour', 'Green cabbage', 'Red grape', 
#     'Wheat bread', 'Chineese plum', 'Sour orange', 'Pita bread', 'Semolina', 'Cape gooseberry', 
#     'Potato bread', 'Hawthorn', 'Black raisin', 'Goji', 'Bulgur', 'Albizia gummifera', 
#     'Monk fruit', 'Acorn squash'
# }

# # Manually assign missing foods to appropriate clusters based on their characteristics
# # Example: Bread products and similar items to processed foods cluster
# processed_foods = {
#     'White bread', 'Bagel', 'Rye bread', 'Raisin bread', 'Cornbread', 'Zwieback', 'Pita bread', 
#     'Taco shell', 'Tostada shell', 'Wonton wrapper', 'Piki bread', 'Rice bread', 'Oat bread', 
#     'Flour', 'Potato bread', 'Semolina', 'Wheat bread'
# }
# for food in processed_foods:
#     presence_clusters_dict.setdefault(11, []).append(food)

# # Other foods assigned to clusters logically based on domain knowledge (e.g., fruits, grains, etc.)
# fruits_and_vegetables = {
#     'Lantern fruit', 'Cantaloupe melon', 'Mikan', 'Green grape', 'Black plum', 'Cape gooseberry', 
#     'Pitaya', 'Green apple', 'Clementine', 'Chineese plum', 'Plantain', 'Yali pear', 'Sour orange', 
#     'Yau choy', 'Iceberg lettuce', 'Green cabbage', 'Juniperus communis', 'Partridge berry', 
#     'Red grape', 'White mulberry', 'Water spinach', 'Pea shoots', 'Mundu', 'Blackberry', 
#     'Rabbiteye blueberry', 'Goji', 'Monk fruit'
# }
# for food in fruits_and_vegetables:
#     presence_clusters_dict.setdefault(4, []).append(food)

# # Specialty items and specific produce
# specialty_items = {'Mate', 'Arabica coffee', 'Robusta coffee', 'Guarana', 'Curry powder', 'Hibiscus tea', 'Miso'}
# for food in specialty_items:
#     presence_clusters_dict.setdefault(7, []).append(food)

# # Move "Arabica coffee" and "Robusta coffee" to Cluster 5 to align with other coffee-related items
# presence_clusters_dict[5].extend(['Arabica coffee', 'Robusta coffee'])

# # Remove them from Cluster 7 if they were added there
# presence_clusters_dict[7] = [food for food in presence_clusters_dict[7] if food not in {'Arabica coffee', 'Robusta coffee'}]


# # Save the revised clusters to a new text file
# def save_clusters_to_file(cluster_dict, file_name):
#     with open(file_name, 'w') as file:
#         for cluster, foods in cluster_dict.items():
#             file.write(f"Cluster {cluster}:\n")
#             for food in foods:
#                 file.write(f"{food}\n")
#             file.write("\n")

# # Save the updated cluster dictionary to a new file
# save_clusters_to_file(presence_clusters_dict, 'C:/Users/labro/Downloads/revised_reorganized_clusters.txt')

# Start by refining the cluster assignments from the uploaded file
# Adjust clusters based on logical groupings, semantic meaning, and coherence

output_path = 'C:/Users/labro/Downloads/Thesis_Food/compounds_presence/average_linkage_clusters.txt'

# Read the file
with open(output_path, 'r') as f:
    clusters_text = f.read()

# Parse the clusters into a structured format
clusters = {}
current_cluster = None

for line in clusters_text.splitlines():
    if line.startswith("Cluster"):
        current_cluster = line
        clusters[current_cluster] = []
    elif line.strip():  # Non-empty line
        clusters[current_cluster].append(line.strip())

# Flatten the clusters to count total foods
all_foods = [food for foods in clusters.values() for food in foods]
total_food_count = len(all_foods)

reorganized_clusters = {
    "Sweet Fruits": [],
    "Bitter Vegetables and Greens": [],
    "Umami-Rich Meats": [],
    "Poultry and Game Birds": [],
    "Seafood - Fish": [],
    "Seafood - Shellfish and Mollusks": [],
    "Grains and Grain Products": [],
    "Legumes": [],
    "Nuts and Seeds": [],
    "Dairy and Alternatives": [],
    "Mushrooms and Fungi": [],
    "Herbs and Spices": [],
    "Oils and Fats": [],
    "Beverages": [],
    "Fermented Foods": [],
    "Miscellaneous": [],
}

# Define reorganizing logic based on current categories
for cluster_id, foods in clusters.items():
    for food in foods:
        # Assign foods to clusters based on known species/flavors
        if any(term in food.lower() for term in ["fruit", "berry", "apple", "melon"]):
            reorganized_clusters["Sweet Fruits"].append(food)
        elif any(term in food.lower() for term in ["lettuce", "broccoli", "kale", "greens", "chard", "bitter"]):
            reorganized_clusters["Bitter Vegetables and Greens"].append(food)
        elif any(term in food.lower() for term in ["beef", "meat", "boar", "venison", "bison", "goat"]):
            reorganized_clusters["Umami-Rich Meats"].append(food)
        elif any(term in food.lower() for term in ["chicken", "turkey", "duck", "poultry", "pheasant"]):
            reorganized_clusters["Poultry and Game Birds"].append(food)
        elif any(term in food.lower() for term in ["fish", "salmon", "cod", "trout", "tuna"]):
            reorganized_clusters["Seafood - Fish"].append(food)
        elif any(term in food.lower() for term in ["crab", "shrimp", "lobster", "clam", "scallop"]):
            reorganized_clusters["Seafood - Shellfish and Mollusks"].append(food)
        elif any(term in food.lower() for term in ["bread", "grain", "rice", "flour", "pasta", "cereal"]):
            reorganized_clusters["Grains and Grain Products"].append(food)
        elif any(term in food.lower() for term in ["bean", "lentil", "pea", "legume"]):
            reorganized_clusters["Legumes"].append(food)
        elif any(term in food.lower() for term in ["nut", "seed", "almond", "walnut", "pistachio"]):
            reorganized_clusters["Nuts and Seeds"].append(food)
        elif any(term in food.lower() for term in ["milk", "cheese", "yogurt", "cream", "dairy"]):
            reorganized_clusters["Dairy and Alternatives"].append(food)
        elif any(term in food.lower() for term in ["mushroom", "fungi", "truffle"]):
            reorganized_clusters["Mushrooms and Fungi"].append(food)
        elif any(term in food.lower() for term in ["herb", "spice", "oregano", "cinnamon", "mint"]):
            reorganized_clusters["Herbs and Spices"].append(food)
        elif any(term in food.lower() for term in ["oil", "fat", "butter"]):
            reorganized_clusters["Oils and Fats"].append(food)
        elif any(term in food.lower() for term in ["tea", "coffee", "beverage", "wine"]):
            reorganized_clusters["Beverages"].append(food)
        elif any(term in food.lower() for term in ["fermented", "miso", "kimchi", "sauerkraut"]):
            reorganized_clusters["Fermented Foods"].append(food)
        else:
            reorganized_clusters["Miscellaneous"].append(food)


# Prepare refined clusters for saving and sharing
refined_clusters_df = pd.DataFrame({
    "Cluster": list(reorganized_clusters.keys()),
    "Food Count": [len(foods) for foods in reorganized_clusters.values()],
    "Foods": [", ".join(foods) for foods in reorganized_clusters.values()]
})

total_refined_food_count = sum(len(foods) for foods in reorganized_clusters.values())
print(total_refined_food_count)

# Save refined clusters to file
refined_file_path = 'C:/Users/labro/Downloads/reorganized_clusters_refined.csv'
refined_clusters_df.to_csv(refined_file_path, index=False)
# processed_sub_clusters_df.to_csv(refined_file_path, index=False)
