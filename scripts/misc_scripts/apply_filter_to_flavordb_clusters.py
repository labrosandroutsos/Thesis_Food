import pandas as pd
import re # Import regex module

def filter_clusters_by_food_list(cluster_file_path, food_list_csv_path, output_file_path):
    """
    Filters a cluster file, keeping only foods present in a given food list CSV.
    Handles food names containing commas.

    Args:
        cluster_file_path (str): Path to the input cluster file (.txt).
        food_list_csv_path (str): Path to the CSV file containing allowed food names in the index.
        output_file_path (str): Path to save the filtered cluster file (.txt).
    """
    try:
        # Load the allowed food names from the CSV index
        food_df = pd.read_csv(food_list_csv_path, sep=';', index_col=0)
        allowed_foods = set(food_df.index)
        print(f"Loaded {len(allowed_foods)} allowed food names.")

        filtered_clusters = {}
        current_cluster_name = None
        current_foods_list = [] # Store foods for the current cluster

        # Read and parse the cluster file
        with open(cluster_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Cluster'):
                    # Process the previous cluster's foods before starting a new one
                    if current_cluster_name and current_foods_list:
                        cluster_content = ", ".join(current_foods_list) # Join foods collected for the cluster
                        found_foods_in_cluster = []
                        for food in allowed_foods:
                            # Use regex to find whole food names, considering boundaries (comma, start/end)
                            # This handles food names with commas inside them
                            pattern = r'(?:^|,\s*)' + re.escape(food) + r'(?:\s*,|$)'
                            if re.search(pattern, cluster_content):
                                found_foods_in_cluster.append(food)
                        if found_foods_in_cluster:
                             filtered_clusters[current_cluster_name] = sorted(found_foods_in_cluster)

                    # Start new cluster
                    current_cluster_name = line[:-1] # Remove trailing ':'
                    current_foods_list = [] # Reset food list for the new cluster
                elif current_cluster_name:
                    # Append foods from the current line to the list for this cluster
                    # Split by comma, but handle potential extra spaces
                    foods_on_line = [food.strip() for food in line.split(',') if food.strip()]
                    current_foods_list.extend(foods_on_line)

            # Process the last cluster after the loop ends
            if current_cluster_name and current_foods_list:
                cluster_content = ", ".join(current_foods_list)
                found_foods_in_cluster = []
                for food in allowed_foods:
                    pattern = r'(?:^|,\s*)' + re.escape(food) + r'(?:\s*,|$)'
                    if re.search(pattern, cluster_content):
                        found_foods_in_cluster.append(food)
                if found_foods_in_cluster:
                    filtered_clusters[current_cluster_name] = sorted(found_foods_in_cluster)

        # Write the filtered clusters to the output file and track found foods
        total_foods_written = 0
        foods_found_in_clusters = set() # Set to track foods actually written
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # Sort cluster names for consistent output order
            for cluster_name in sorted(filtered_clusters.keys()):
                foods = filtered_clusters[cluster_name]
                outfile.write(f"{cluster_name}:\n")
                outfile.write(', '.join(foods) + '\n\n') # Foods are already sorted
                total_foods_written += len(foods) # Increment the counter
                foods_found_in_clusters.update(foods) # Add written foods to the set

        print(f"Filtered clusters saved to {output_file_path}")
        print(f"Total number of food items in the filtered file: {total_foods_written}")

        # Check for foods that were allowed but not found in any cluster
        missing_foods = allowed_foods - foods_found_in_clusters
        if missing_foods:
            print(f"\nWarning: {len(missing_foods)} food(s) from the allowed list were not found in the input cluster file:")
            for food in sorted(list(missing_foods)):
                print(f"- {food}")
        else:
            print("\nAll foods from the allowed list were found and included in the filtered clusters.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Configuration ---
cluster_input_path = 'clusters/flavordb_clusters/filtered_flavordb_clusters.txt'
food_list_input_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/compounds_presence/foodname_compound_presence_0_1_filtered.csv'
filtered_output_path = 'clusters/flavordb_clusters/filtered_flavordb_clusters_683.txt' # Naming based on expected food count

# --- Run the filtering ---
filter_clusters_by_food_list(cluster_input_path, food_list_input_path, filtered_output_path) 