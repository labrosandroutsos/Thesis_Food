import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import ast

# --- Configuration ---
VECTOR_SIZE = 100  # Desired dimensionality of the embedding vectors
# Node2Vec parameters (can be tuned)
P = 1  # Return parameter
Q = 1  # In-out parameter
NUM_WALKS = 10  # Number of walks per node
WALK_LENGTH = 80  # Length of each walk
WINDOW = 10  # Context window size for Word2Vec (used by Node2Vec)
MIN_COUNT = 1  # Minimal count of words to be included
WORKERS = 12 # Number of parallel workers

# Input data paths (assuming similar structure to word_embeddings.py)
DATA_PATH_TEMPLATE = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/ordered_compounds/ordered_compounds_per_food_{taste}.csv'
TASTES = ['bitter', 'sweet', 'umami', 'other']

# Output path
EMBEDDING_OUTPUT_PATH_TEMPLATE = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings_data/node2vec_food_embeddings_vs{vector_size}_p{p}_q{q}.pkl'

def load_and_prepare_data(data_path_template, tastes):
    """
    Loads data for all tastes, extracts food-compound pairs.
    Returns a list of (food_name, compound_name) edges.
    """
    all_edges = set() # Use a set to avoid duplicate edges if a compound is in multiple taste lists for the same food (though unlikely with current data structure)
    food_nodes = set()
    compound_nodes = set()

    for taste in tastes:
        file_path = data_path_template.format(taste=taste)
        try:
            df = pd.read_csv(file_path, sep=';')
            df['sorted_compounds'] = df['sorted_compounds'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            )
            for _, row in df.iterrows():
                food_name = str(row['food_name']).strip()
                if not food_name:
                    print(f"Warning: Empty food name found in {file_path}, row: {_}")
                    continue
                food_nodes.add(food_name)
                for compound in row['sorted_compounds']:
                    compound_name = str(compound).strip()
                    if not compound_name:
                        # print(f"Warning: Empty compound name for food '{food_name}' in {file_path}")
                        continue
                    all_edges.add((food_name, compound_name))
                    compound_nodes.add(compound_name)
        except FileNotFoundError:
            print(f"Warning: File not found {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"Total unique food nodes: {len(food_nodes)}")
    print(f"Total unique compound nodes: {len(compound_nodes)}")
    print(f"Total unique food-compound edges: {len(all_edges)}")
    return list(all_edges), list(food_nodes)

def create_graph(edges):
    """
    Creates a NetworkX graph from the list of edges.
    """
    G = nx.Graph()
    # Ensure all nodes involved in edges are added, even if some nodes might not have edges
    # (though our edge collection method should mean all nodes in edges are covered)
    unique_nodes = set()
    for u, v in edges:
        unique_nodes.add(u)
        unique_nodes.add(v)
    G.add_nodes_from(list(unique_nodes))
    G.add_edges_from(edges)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def generate_node2vec_embeddings(graph, vector_size, p_param, q_param, num_walks, walk_length, window_size, min_count, workers):
    """
    Generates node embeddings using Node2Vec.
    """
    print("Initializing Node2Vec...")
    # Precompute probabilities and generate walks - Currently, the node2vec library uses gensim's Word2Vec for the skip-gram part.
    node2vec = Node2Vec(graph, dimensions=vector_size, walk_length=walk_length, num_walks=num_walks, p=p_param, q=q_param, workers=workers, quiet=False)

    print("Training Word2Vec model (skip-gram) on walks...")
    # The 'window' and 'min_count' parameters are for the underlying Word2Vec model.
    # The `iter` parameter in `fit` corresponds to epochs for Word2Vec.
    model = node2vec.fit(window=window_size, min_count=min_count, batch_words=4, epochs=1) # Default epochs is 1, can be increased

    print("Embeddings generated.")
    return model

def save_food_embeddings(model, food_nodes, output_path):
    """
    Extracts embeddings for food nodes and saves them to a pickle file.
    """
    food_embeddings = {}
    for food_name in food_nodes:
        if food_name in model.wv:
            food_embeddings[food_name] = model.wv[food_name]
        else:
            print(f"Warning: Food node '{food_name}' not found in Node2Vec model's vocabulary. Skipping.")
            # Optionally, assign a zero vector or handle as per requirement
            # food_embeddings[food_name] = np.zeros(model.wv.vector_size)


    if not food_embeddings:
        print("Error: No food embeddings were extracted. Check food node names and model vocabulary.")
        return

    df_food_embeddings = pd.DataFrame.from_dict(food_embeddings, orient='index')
    # Reset index to make food names a column
    df_food_embeddings = df_food_embeddings.reset_index().rename(columns={'index': 'food_name'})

    # Create columns for each dimension of the embedding
    embedding_cols = [f'embedding_dim_{i}' for i in range(df_food_embeddings.iloc[0].drop('food_name').shape[0])] # df_food_embeddings.iloc[0][1].shape[0])]
    
    # Create a new DataFrame with separate columns for each embedding dimension
    embeddings_split_df = pd.DataFrame(df_food_embeddings.drop(columns=['food_name']).values.tolist(), index=df_food_embeddings.index, columns=embedding_cols)
    final_df = pd.concat([df_food_embeddings[['food_name']], embeddings_split_df], axis=1)

    final_df.to_pickle(output_path)
    print(f"Food embeddings saved to {output_path}")
    print(f"Shape of saved embeddings DataFrame: {final_df.shape}")
    print(final_df.head())


def main():
    print("Starting graph embedding generation process...")

    print("Step 1: Loading and preparing data...")
    edges, food_nodes = load_and_prepare_data(DATA_PATH_TEMPLATE, TASTES)

    if not edges:
        print("No edges found. Cannot proceed with graph creation. Exiting.")
        return
    if not food_nodes:
        print("No food nodes identified. Cannot proceed to save food-specific embeddings. Exiting.")
        return

    print("Step 2: Creating graph...")
    graph = create_graph(edges)

    if graph.number_of_nodes() == 0:
        print("Graph is empty. Cannot generate embeddings. Exiting.")
        return

    output_path = EMBEDDING_OUTPUT_PATH_TEMPLATE.format(vector_size=VECTOR_SIZE, p=P, q=Q)

    print(f"\nStep 3: Generating Node2Vec embeddings (vector_size={VECTOR_SIZE}, p={P}, q={Q})...")
    node2vec_model = generate_node2vec_embeddings(graph,
                                                  vector_size=VECTOR_SIZE,
                                                  p_param=P,
                                                  q_param=Q,
                                                  num_walks=NUM_WALKS,
                                                  walk_length=WALK_LENGTH,
                                                  window_size=WINDOW,
                                                  min_count=MIN_COUNT,
                                                  workers=WORKERS)

    print("\nStep 4: Extracting and saving food embeddings...")
    save_food_embeddings(node2vec_model, food_nodes, output_path)

    print("\nProcess completed.")

if __name__ == '__main__':
    main() 