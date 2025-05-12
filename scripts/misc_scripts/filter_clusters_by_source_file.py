import re
import os
from collections import defaultdict

def split_foods(line):
    # Use a regular expression to split on commas that are not within parentheses
    # Handles cases like "Apple, Gala" or "Bean (Phaseolus vulgaris), Kidney"
    return re.split(r',\s*(?![^()]*\))', line)

def load_foods_from_cluster_file(filepath):
    """Loads all unique food names from a cluster file."""
    allowed_foods = set()
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line or line.startswith('Cluster'):
                    continue
                foods_on_line = [food.strip() for food in split_foods(line) if food.strip()]
                allowed_foods.update(foods_on_line)
        print(f"Loaded {len(allowed_foods)} unique allowed foods from {filepath}")
        return allowed_foods
    except Exception as e:
        print(f"Error reading source file {filepath}: {e}")
        return None

def filter_target_cluster_file(target_filepath, output_filepath, allowed_foods_set):
    """Filters the target cluster file, keeping only allowed foods."""
    if not os.path.exists(target_filepath):
        print(f"Error: Target file not found at {target_filepath}")
        return

    if allowed_foods_set is None:
        print("Error: No allowed foods loaded. Cannot filter.")
        return
        
    try:
        current_cluster_header = None
        kept_foods_for_cluster = []
        total_foods_written = 0
        clusters_written = 0

        with open(target_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if not line: # Skip empty lines but handle potential cluster breaks
                    # Write previous cluster if it had content and wasn't just an empty line break
                    if current_cluster_header and kept_foods_for_cluster:
                         outfile.write(f"{current_cluster_header}\n")
                         outfile.write(', '.join(sorted(kept_foods_for_cluster)) + '\n\n')
                         total_foods_written += len(kept_foods_for_cluster)
                         clusters_written += 1
                    # Reset for potential next cluster, even if separated by blank lines
                    current_cluster_header = None 
                    kept_foods_for_cluster = []
                    continue # Move to next line

                if line.startswith('Cluster'):
                    # Write the *previous* cluster's filtered foods if any were kept
                    if current_cluster_header and kept_foods_for_cluster:
                        outfile.write(f"{current_cluster_header}\n")
                        outfile.write(', '.join(sorted(kept_foods_for_cluster)) + '\n\n')
                        total_foods_written += len(kept_foods_for_cluster)
                        clusters_written += 1
                        
                    # Start the new cluster
                    current_cluster_header = line 
                    kept_foods_for_cluster = [] # Reset list for new cluster
                
                elif current_cluster_header: # Process food lines only if we are inside a cluster
                    foods_on_line = [food.strip() for food in split_foods(line) if food.strip()]
                    # Filter this line's foods
                    filtered_line_foods = [food for food in foods_on_line if food in allowed_foods_set]
                    kept_foods_for_cluster.extend(filtered_line_foods)

            # --- After the loop: Write the last cluster ---
            if current_cluster_header and kept_foods_for_cluster:
                outfile.write(f"{current_cluster_header}\n")
                outfile.write(', '.join(sorted(kept_foods_for_cluster)) + '\n\n')
                total_foods_written += len(kept_foods_for_cluster)
                clusters_written += 1

        print(f"\nFiltered cluster file saved to: {output_filepath}")
        print(f"Total clusters written: {clusters_written}")
        print(f"Total food items written: {total_foods_written}")

    except Exception as e:
        print(f"Error processing target file {target_filepath} or writing output: {e}")


if __name__ == "__main__":
    print("--- Cluster File Filtering Script ---")
    
    source_file=('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/misc_clusters/expanded_refined_food_clusters_683_CHATGPT_CLUSTERS.txt')  # 683 foods clusters

    target_file = ("/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/misc_clusters/final_fully_coherent_clusters_962_foodb_Foods.txt")
    output_file = ("/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/misc_clusters/initial_clusters_filtered_683_foods.txt")

    allowed_foods = load_foods_from_cluster_file(source_file)

    if allowed_foods:
        filter_target_cluster_file(target_file, output_file, allowed_foods) 