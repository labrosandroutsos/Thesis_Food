from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import ast 

# Load data
df_food_dict_bitter = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_bitter.csv', sep=';')
df_food_dict_sweet = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_sweet.csv', sep=';')
df_food_dict_umami = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_umami.csv', sep=';')
df_food_dict_other = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_other.csv', sep=';')

# Prepare the data: List of lists (sentences) for each taste
def prepare_sentences(df):
    df['sorted_compounds'] = df['sorted_compounds'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df['sorted_compounds'].tolist()

sentences_bitter = prepare_sentences(df_food_dict_bitter)
sentences_sweet = prepare_sentences(df_food_dict_sweet)
sentences_umami = prepare_sentences(df_food_dict_umami)
sentences_other = prepare_sentences(df_food_dict_other)

def get_compound_embeddings(model, compounds):
    """Get embeddings for each compound in sorted_compounds"""
    embeddings = []
    for compound in compounds:
        if compound in model.wv:
            embeddings.append(model.wv[compound])
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)

def train_models_and_get_embeddings(sentences_dict, embedding_size, window_size):
    """Train models and get embeddings for all tastes with given parameters"""
    embeddings_dict = {}
    
    for taste, sentences in sentences_dict.items():
        # Train model
        model = Word2Vec(sentences, vector_size=embedding_size, window=window_size, 
                        min_count=1, workers=6)
        
        # Get embeddings for the corresponding dataframe
        df_name = f'df_food_dict_{taste}'
        df = globals()[df_name]  # Get the dataframe from global namespace
        embeddings = df['sorted_compounds'].apply(lambda x: get_compound_embeddings(model, x))
        embeddings_dict[taste] = embeddings
    
    return embeddings_dict

def concatenate_embeddings(row, vector_size):
    """Concatenate and aggregate embeddings from different tastes"""
    # Extract embeddings from the row
    bitter_embedding = row['compound_embeddings_bitter']
    sweet_embedding = row['compound_embeddings_sweet']
    umami_embedding = row['compound_embeddings_umami']
    other_embedding = row['compound_embeddings_other']
    
    # Initialize accumulators for each taste embedding
    food_bitter = np.zeros(vector_size)
    food_sweet = np.zeros(vector_size)
    food_umami = np.zeros(vector_size)
    food_other = np.zeros(vector_size)

    # Sum over each compound for each taste embedding
    for i in range(bitter_embedding.shape[0]):
        food_bitter += bitter_embedding[i]
        food_sweet += sweet_embedding[i]
        food_umami += umami_embedding[i]
        food_other += other_embedding[i]
    
    # Concatenate all taste embeddings
    return np.concatenate((food_bitter, food_sweet, food_umami, food_other), axis=None)

def create_unified_embeddings(embedding_size, window_size):
    """Create unified embeddings for given parameters"""
    print(f"\nProcessing: embedding_size={embedding_size}, window_size={window_size}")
    
    # Train models and get embeddings for all tastes
    sentences_dict = {
        'bitter': sentences_bitter,
        'sweet': sentences_sweet,
        'umami': sentences_umami,
        'other': sentences_other
    }
    
    embeddings_dict = train_models_and_get_embeddings(sentences_dict, embedding_size, window_size)
    
    # Create DataFrames with embeddings
    dfs = {}
    for taste in sentences_dict.keys():
        df_name = f'df_food_dict_{taste}'
        df = globals()[df_name].copy()
        df[f'compound_embeddings_{taste}'] = embeddings_dict[taste]
        dfs[taste] = df[['food_name', f'compound_embeddings_{taste}']]
    
    # Merge all DataFrames
    merged_df = dfs['bitter']
    for taste in ['sweet', 'umami', 'other']:
        merged_df = pd.merge(merged_df, dfs[taste], on='food_name')
    
    # Create unified embeddings
    merged_df['unified_embedding'] = merged_df.apply(
        lambda row: concatenate_embeddings(row, embedding_size), axis=1)
    
    # Create final DataFrame
    final_df = merged_df[['food_name', 'unified_embedding']]
    
    # Save the unified embeddings
    output_path = f'C:/Users/labro/Downloads/Thesis_Food/final_unified_embeddings_{embedding_size}_{window_size}.pkl'
    final_df.to_pickle(output_path)
    print(f"Saved unified embeddings to: {output_path}")
    
    return final_df

# Configuration for different models
embedding_sizes = [50, 100, 150, 200, 250]
window_sizes = [2, 3, 5, 7, 10]

# Process all combinations
for embedding_size in embedding_sizes:
    for window_size in window_sizes:
        create_unified_embeddings(embedding_size, window_size)

print("\nAll embeddings have been generated and unified!")
