from gensim.models import Word2Vec, FastText
import pandas as pd
import numpy as np
import ast 
import argparse
import os

def concatenate_embeddings(row, vector_size):
    vector_size = vector_size
    # Extract embeddings from the row and concatenate them
    # print(type(row['compound_embeddings_bitter']))
    # print('row', row)
    bitter_embedding = (row['compound_embeddings_bitter'])
    # print('bitter row shape', bitter_embedding.shape)
    sweet_embedding = (row['compound_embeddings_sweet'])
    umami_embedding = (row['compound_embeddings_umami'])
    other_embedding = (row['compound_embeddings_other'])
    

    # Initialize accumulators for each taste embedding
    food_bitter = np.zeros(vector_size)
    food_sweet = np.zeros(vector_size)
    food_umami = np.zeros(vector_size)
    food_other = np.zeros(vector_size)

    # print("bitter_embedding.shape =", bitter_embedding.shape)

    # Sum over each compound for each taste embedding
    num_compounds = bitter_embedding.shape[0]  # Assuming all taste embeddings have the same number of compounds
    for i in range(num_compounds):
        # print("Inside loop:", bitter_embedding[i].shape)

        # Commented out summation
        # food_bitter += bitter_embedding[i]
        # food_sweet += sweet_embedding[i]
        # food_umami += umami_embedding[i]
        # food_other += other_embedding[i]

        # Accumulate sums for averaging
        food_bitter += bitter_embedding[i]
        food_sweet += sweet_embedding[i]
        food_umami += umami_embedding[i]
        food_other += other_embedding[i]

    # Calculate average if num_compounds > 0
    # if num_compounds > 0:
    #     food_bitter /= num_compounds
    #     food_sweet /= num_compounds
    #     food_umami /= num_compounds
    #     food_other /= num_compounds
    # else: the embeddings remain zeros, which is the correct average for zero compounds
    
    food_all = np.concatenate((food_bitter, food_sweet, food_umami, food_other), axis=None)


    # Concatenate embeddings into one unified vector
    # unified_embedding = np.concatenate([bitter_embedding, sweet_embedding, umami_embedding, other_embedding], axis=1)
    # unified_embedding = np.concatenate([row['compound_embeddings_bitter'], row['compound_embeddings_sweet'], row['compound_embeddings_umami'], row['compound_embeddings_other']], axis=1)
    compound_embedding = bitter_embedding + sweet_embedding + umami_embedding + other_embedding
    # print('food all', food_all.shape)
    # return unified_embedding
    return food_all

def get_compound_embeddings(model, compounds):
    embeddings = []
    for compound in compounds:
        # FastText can generate vectors for OOV words, check if needed
        # For consistency with Word2Vec approach (returning zeros), we keep this check.
        # If you want OOV vectors: remove the `if compound in model.wv:` check.
        if compound in model.wv:
            embeddings.append(model.wv[compound])
        else:
            # Optional: Log OOV words
            # print(f"Warning: Compound '{compound}' not in vocabulary.")
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)

def main(sg_flag, negative_samples, epochs_count):
    # Define base paths
    base_path = '/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food'
    ordered_compounds_path = os.path.join(base_path, 'ordered_compounds')
    models_path = os.path.join(base_path, 'models')
    embeddings_path = os.path.join(base_path, 'embeddings_data')
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(embeddings_path, exist_ok=True)
    
    # --- Load Data ---
    print("Loading compound data...")
    df_food_dict_bitter = pd.read_csv(os.path.join(ordered_compounds_path, 'ordered_compounds_per_food_bitter.csv'), sep=';')
    df_food_dict_sweet = pd.read_csv(os.path.join(ordered_compounds_path, 'ordered_compounds_per_food_sweet.csv'), sep=';')
    df_food_dict_umami = pd.read_csv(os.path.join(ordered_compounds_path, 'ordered_compounds_per_food_umami.csv'), sep=';')
    df_food_dict_other = pd.read_csv(os.path.join(ordered_compounds_path, 'ordered_compounds_per_food_other.csv'), sep=';')
    
    # --- Prepare Sentences ---
    print("Preparing sentences...")
    def prepare_sentences(df):
        df['sorted_compounds'] = df['sorted_compounds'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df = df[df['sorted_compounds'].apply(lambda x: isinstance(x, list))]
        return df['sorted_compounds'].tolist()

    sentences_bitter = prepare_sentences(df_food_dict_bitter)
    sentences_sweet = prepare_sentences(df_food_dict_sweet)
    sentences_umami = prepare_sentences(df_food_dict_umami)
    sentences_other = prepare_sentences(df_food_dict_other)
    
    embedding_sizes = [50, 100, 150, 200, 250]
    window_sizes = [2, 3, 5, 7, 10]

    # Create suffix for filenames based on FastText params
    ft_suffix = f"_fasttext_sg{sg_flag}_neg{negative_samples}_ep{epochs_count}"
    print(f"Using FastText parameters: sg={sg_flag}, negative={negative_samples}, epochs={epochs_count}")
    print(f"Filename suffix: {ft_suffix}")

    # Train models with different configurations
    for embedding_size in embedding_sizes:
        for window_size in window_sizes:
            print(f"\nProcessing: embedding_size={embedding_size}, window_size={window_size}")
            
            df_bitter_tmp = df_food_dict_bitter.copy()
            df_sweet_tmp  = df_food_dict_sweet.copy()
            df_umami_tmp  = df_food_dict_umami.copy()
            df_other_tmp  = df_food_dict_other.copy()

            # --- Train FastText models ---
            print(f"  Training FastText models...")
            model_params = {
                'vector_size': embedding_size,
                'window': window_size,
                'min_count': 1, 
                'workers': 8,
                'sg': sg_flag, 
                'negative': negative_samples, 
                'epochs': epochs_count
            }
            # Use FastText class instead of Word2Vec
            model_bitter = FastText(sentences=sentences_bitter, **model_params)
            model_sweet = FastText(sentences=sentences_sweet, **model_params)
            model_umami = FastText(sentences=sentences_umami, **model_params)
            model_other = FastText(sentences=sentences_other, **model_params)

            # --- Save models with configuration in filename ---
            # Update filenames to indicate FastText
            model_bitter.save(os.path.join(models_path, f"fasttext_bitter_{embedding_size}_{window_size}{ft_suffix}.model"))
            model_sweet.save(os.path.join(models_path, f"fasttext_sweet_{embedding_size}_{window_size}{ft_suffix}.model"))
            model_umami.save(os.path.join(models_path, f"fasttext_umami_{embedding_size}_{window_size}{ft_suffix}.model"))
            model_other.save(os.path.join(models_path, f"fasttext_other_{embedding_size}_{window_size}{ft_suffix}.model"))
            print(f"  FastText Models saved.")

            # --- Get compound embeddings for each food ---
            print(f"  Calculating compound embeddings...")
            df_bitter_tmp['compound_embeddings_bitter'] = df_bitter_tmp['sorted_compounds'].apply(lambda x: get_compound_embeddings(model_bitter, x))
            df_sweet_tmp['compound_embeddings_sweet'] = df_sweet_tmp['sorted_compounds'].apply(lambda x: get_compound_embeddings(model_sweet, x))
            df_umami_tmp['compound_embeddings_umami'] = df_umami_tmp['sorted_compounds'].apply(lambda x: get_compound_embeddings(model_umami, x))
            df_other_tmp['compound_embeddings_other'] = df_other_tmp['sorted_compounds'].apply(lambda x: get_compound_embeddings(model_other, x))

            # --- Merging and Aggregating Food Embeddings ---
            print(f"  Merging and aggregating embeddings...")
            merged_df = pd.merge(df_bitter_tmp[['food_name', 'compound_embeddings_bitter']], 
                                df_sweet_tmp[['food_name', 'compound_embeddings_sweet']], on='food_name')
            merged_df = pd.merge(merged_df, df_umami_tmp[['food_name', 'compound_embeddings_umami']], on='food_name')
            merged_df = pd.merge(merged_df, df_other_tmp[['food_name', 'compound_embeddings_other']], on='food_name')

            merged_df['unified_embedding'] = merged_df.apply(concatenate_embeddings, args=(embedding_size,), axis=1)
            
            # --- Final DataFrame and Saving ---
            final_df = merged_df[['food_name', 'unified_embedding']]
            print(f"  Final DataFrame shape: {final_df.shape}")
            
            # Update output filename for embeddings
            output_filename = f"final_unified_embeddings_fasttext_{embedding_size}_{window_size}{ft_suffix}.pkl"
            output_path = os.path.join(embeddings_path, output_filename)
            final_df.to_pickle(output_path)
            print(f"  Embeddings saved to: {output_path}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FastText models and generate food embeddings.')
    parser.add_argument('--sg', type=int, default=0, choices=[0, 1],
                        help='FastText training algorithm: 0 for CBOW, 1 for skip-gram (default: 0)')
    parser.add_argument('--negative', type=int, default=5,
                        help='Number of negative samples for FastText (default: 5)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs for FastText (default: 5)')
    
    args = parser.parse_args()
    
    main(sg_flag=args.sg, negative_samples=args.negative, epochs_count=args.epochs)

# keep the first ten rows
# import ast
# df_food_dict_other['sorted_compounds'] = df_food_dict_other['sorted_compounds'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# first_compound_abalone = df_food_dict_other.loc[df_food_dict_other['food_name'] == 'Abalone', 'sorted_compounds'].values[0][0]
# first_compound_acorn = df_food_dict_other.loc[df_food_dict_other['food_name'] == 'Acorn squash', 'sorted_compounds'].values[0][0]

# print("First compound for Abalone:", first_compound_abalone)
# print("First compound for Acorn squash:", first_compound_acorn)

# # Print their embeddings
# if first_compound_abalone in model_other.wv and first_compound_acorn in model_other.wv:
#     print("Embedding for first compound in Abalone:", model_other.wv[first_compound_abalone])
#     print("Embedding for first compound in Acorn squash:", model_other.wv[first_compound_acorn])
# else:
#     print("One or both compounds are not found in the Word2Vec vocabulary.")

# # Check the size of the Word2Vec vocabulary
# print(f"Vocabulary size: {len(model_other.wv.key_to_index)}")

# Check a few random compound embeddings
# random_compounds = list(model_other.wv.key_to_index.keys())[:5]  # Get 5 random compounds from the vocabulary
# for compound in random_compounds:
#     print(f"Embedding for {compound}: {model_other.wv[compound]}")

# df_food_dict_other = df_food_dict_other.head(10)
# df_food_dict_other['taste_embedding'] = df_food_dict_other['sorted_compounds'].apply(lambda x: get_compound_embeddings(model_other, x))

# check if the first 5 rows have the same value at taste_embedding column.. Use equator =
# print(df_food_dict_other['taste_embedding'][0] == df_food_dict_other['taste_embedding'][1] == df_food_dict_other['taste_embedding'][2] == df_food_dict_other['taste_embedding'][3] == df_food_dict_other['taste_embedding'][4])

# model_other.save("C:/Users/labro/Downloads/Thesis_Food/models/word2vec_other_correct_150.model")
# df_food_dict_other.to_csv('C:/Users/labro/Downloads/Thesis_Food/embeddings/df_food_dict_other_embeddings_correct.csv', index=False, sep=';')

# df_food_dict_other[['food_name', 'taste_embedding']].to_pickle('embedding_data_other_150.pkl')

# Save the Word2Vec models for each taste

# Save each DataFrame with embeddings to a CSV file

# Now each food has a list of embeddings (one for each compound)
# print(df_food_dict[['food_name', 'compound_embeddings']])