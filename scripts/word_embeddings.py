from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import ast 

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
    for i in range(bitter_embedding.shape[0]):
        # print("Inside loop:", bitter_embedding[i].shape)

        food_bitter += bitter_embedding[i]
        food_sweet += sweet_embedding[i]
        food_umami += umami_embedding[i]
        food_other += other_embedding[i]
    
    food_all = np.concatenate((food_bitter, food_sweet, food_umami, food_other), axis=None)


    # Concatenate embeddings into one unified vector
    # unified_embedding = np.concatenate([bitter_embedding, sweet_embedding, umami_embedding, other_embedding], axis=1)
    # unified_embedding = np.concatenate([row['compound_embeddings_bitter'], row['compound_embeddings_sweet'], row['compound_embeddings_umami'], row['compound_embeddings_other']], axis=1)
    compound_embedding = bitter_embedding + sweet_embedding + umami_embedding + other_embedding
    # print('food all', food_all.shape)
    # return unified_embedding
    return food_all


df_food_dict_bitter = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_bitter.csv', sep=';')
df_food_dict_sweet = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_sweet.csv', sep=';')
df_food_dict_umami = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_umami.csv', sep=';')
df_food_dict_other = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds/ordered_compounds_per_food_other.csv', sep=';')
# print(df_food_dict_other[['food_name', 'sorted_compounds']].head())

# Prepare the data: List of lists (sentences) for each taste
df_food_dict_bitter['sorted_compounds'] = df_food_dict_bitter['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
sentences_bitter = df_food_dict_bitter['sorted_compounds'].tolist()

df_food_dict_sweet['sorted_compounds'] = df_food_dict_sweet['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
sentences_sweet = df_food_dict_sweet['sorted_compounds'].tolist()

df_food_dict_umami['sorted_compounds'] = df_food_dict_umami['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
sentences_umami = df_food_dict_umami['sorted_compounds'].tolist()

df_food_dict_other['sorted_compounds'] = df_food_dict_other['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
sentences_other = df_food_dict_other['sorted_compounds'].tolist()
# print((sentences_other[:1]))
# lenth of the sentences


# Step 2: Get embeddings for each compound in sorted_compounds
def get_compound_embeddings(model, compounds):
    embeddings = []
    for compound in compounds:
        if compound in model.wv:
            embeddings.append(model.wv[compound])
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)
    
# Configuration for different models
# embedding_sizes = [50, 100, 150, 200, 250]
embedding_sizes = [100, 150, 200, 250]
window_sizes = [2, 3, 5, 7, 10]

# Train models with different configurations
for embedding_size in embedding_sizes:
    for window_size in window_sizes:
        if embedding_size == 100 and window_size == 2:
            continue
        elif embedding_size == 100 and window_size == 3:
            continue

        df_bitter_tmp = df_food_dict_bitter.copy()
        df_sweet_tmp  = df_food_dict_sweet.copy()
        df_umami_tmp  = df_food_dict_umami.copy()
        df_other_tmp  = df_food_dict_other.copy()
        print(f"\nTraining models with embedding_size={embedding_size}, window_size={window_size}")
        
        # Train Word2Vec models for each taste with current configuration
        model_bitter = Word2Vec(sentences_bitter, vector_size=embedding_size, window=window_size, min_count=1, workers=6)
        model_sweet = Word2Vec(sentences_sweet, vector_size=embedding_size, window=window_size, min_count=1, workers=6)
        model_umami = Word2Vec(sentences_umami, vector_size=embedding_size, window=window_size, min_count=1, workers=6)
        model_other = Word2Vec(sentences_other, vector_size=embedding_size, window=window_size, min_count=1, workers=6)

        # Save models with configuration in filename
        model_bitter.save(f"C:/Users/labro/Downloads/Thesis_Food/models/word2vec_bitter_{embedding_size}_{window_size}.model")
        model_sweet.save(f"C:/Users/labro/Downloads/Thesis_Food/models/word2vec_sweet_{embedding_size}_{window_size}.model")
        model_umami.save(f"C:/Users/labro/Downloads/Thesis_Food/models/word2vec_umami_{embedding_size}_{window_size}.model")
        model_other.save(f"C:/Users/labro/Downloads/Thesis_Food/models/word2vec_other_{embedding_size}_{window_size}.model")


        if 'compound_embeddings_bitter' in df_bitter_tmp.columns:
            df_bitter_tmp.drop(columns='compound_embeddings_bitter', inplace=True)
        if 'compound_embeddings_sweet' in df_sweet_tmp.columns:
            df_sweet_tmp.drop(columns='compound_embeddings_sweet', inplace=True)
        if 'compound_embeddings_umami' in df_umami_tmp.columns:
            df_umami_tmp.drop(columns='compound_embeddings_umami', inplace=True)
        if 'compound_embeddings_other' in df_other_tmp.columns:
            df_other_tmp.drop(columns='compound_embeddings_other', inplace=True)

        # Get embeddings for each taste
        df_bitter_tmp['taste_embedding'] = df_bitter_tmp['sorted_compounds'].apply(
            lambda x: get_compound_embeddings(model_bitter, x))
        df_sweet_tmp['taste_embedding'] = df_sweet_tmp['sorted_compounds'].apply(
            lambda x: get_compound_embeddings(model_sweet, x))
        df_umami_tmp['taste_embedding'] = df_umami_tmp['sorted_compounds'].apply(
            lambda x: get_compound_embeddings(model_umami, x))
        df_other_tmp['taste_embedding'] = df_other_tmp['sorted_compounds'].apply(
            lambda x: get_compound_embeddings(model_other, x))

        # Save embeddings with configuration in filename
        # df_food_dict_bitter[['food_name', 'taste_embedding']].to_pickle(
        #     f'embedding_data_bitter_{embedding_size}_{window_size}.pkl')
        # df_food_dict_sweet[['food_name', 'taste_embedding']].to_pickle(
        #     f'embedding_data_sweet_{embedding_size}_{window_size}.pkl')
        # df_food_dict_umami[['food_name', 'taste_embedding']].to_pickle(
        #     f'embedding_data_umami_{embedding_size}_{window_size}.pkl')
        # df_food_dict_other[['food_name', 'taste_embedding']].to_pickle(
        #     f'embedding_data_other_{embedding_size}_{window_size}.pkl')

        df_bitter_tmp = df_bitter_tmp.rename(columns={'taste_embedding': 'compound_embeddings_bitter'})
        df_sweet_tmp = df_sweet_tmp.rename(columns={'taste_embedding': 'compound_embeddings_sweet'})
        df_umami_tmp = df_umami_tmp.rename(columns={'taste_embedding': 'compound_embeddings_umami'})
        df_other_tmp = df_other_tmp.rename(columns={'taste_embedding': 'compound_embeddings_other'})


        # Merging the four DataFrames on 'food_name'
        merged_df = pd.merge(df_bitter_tmp[['food_name', 'compound_embeddings_bitter']], 
                            df_sweet_tmp[['food_name', 'compound_embeddings_sweet']], 
                            on='food_name')

        merged_df = pd.merge(merged_df, df_umami_tmp[['food_name', 'compound_embeddings_umami']], 
                            on='food_name')

        merged_df = pd.merge(merged_df, df_other_tmp[['food_name', 'compound_embeddings_other']], 
                            on='food_name')

        merged_df['aggregated_compound_embedding'] = merged_df.apply(concatenate_embeddings,  args=(embedding_size,), axis=1)
        # Rename the final embedding column for clarity
        food_embeddings = merged_df.rename(columns={'aggregated_compound_embedding': 'unified_embedding'})

        # # # Step 4: Final DataFrame with food_name and unified embeddings
        final_df = food_embeddings[['food_name', 'unified_embedding']]

        # # # # Print or Save the Final Unified DataFrame
        print(final_df.head)
        print(final_df.shape)

        final_df.to_pickle(f'C:/Users/labro/Downloads/Thesis_Food/embeddings_data/final_unified_embeddings_aggregated_{embedding_size}_{window_size}.pkl')



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