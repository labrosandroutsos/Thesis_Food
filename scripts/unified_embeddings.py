import pandas as pd
import numpy as np

df_food_dict_bitter = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/embedding_data_bitter_250.pkl')
df_food_dict_sweet = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/embedding_data_sweet_250.pkl')
df_food_dict_umami = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/embedding_data_umami_250.pkl')
df_food_dict_other = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/embedding_data_other_250.pkl')

df_food_dict_bitter = df_food_dict_bitter.rename(columns={'taste_embedding': 'compound_embeddings_bitter'})
df_food_dict_sweet = df_food_dict_sweet.rename(columns={'taste_embedding': 'compound_embeddings_sweet'})
df_food_dict_umami = df_food_dict_umami.rename(columns={'taste_embedding': 'compound_embeddings_umami'})
df_food_dict_other = df_food_dict_other.rename(columns={'taste_embedding': 'compound_embeddings_other'})

# Merging the four DataFrames on 'food_name'
merged_df = pd.merge(df_food_dict_bitter[['food_name', 'compound_embeddings_bitter']], 
                     df_food_dict_sweet[['food_name', 'compound_embeddings_sweet']], 
                     on='food_name')

merged_df = pd.merge(merged_df, df_food_dict_umami[['food_name', 'compound_embeddings_umami']], 
                     on='food_name')

merged_df = pd.merge(merged_df, df_food_dict_other[['food_name', 'compound_embeddings_other']], 
                     on='food_name')

def concatenate_embeddings(row):
    vector_size = 250
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

    # Sum over each compound for each taste embedding
    for i in range(bitter_embedding.shape[0]):
        food_bitter += bitter_embedding[i]
        food_sweet += sweet_embedding[i]
        food_umami += umami_embedding[i]
        food_other += other_embedding[i]
    
    food_all = np.concatenate((food_bitter, food_sweet, food_umami, food_other), axis=None)


    # Concatenate embeddings into one unified vector
    # unified_embedding = np.concatenate([bitter_embedding, sweet_embedding, umami_embedding, other_embedding], axis=1)
    # unified_embedding = np.concatenate([row['compound_embeddings_bitter'], row['compound_embeddings_sweet'], row['compound_embeddings_umami'], row['compound_embeddings_other']], axis=1)
    compound_embedding = bitter_embedding + sweet_embedding + umami_embedding + other_embedding
    print('food all', food_all.shape)
    # return unified_embedding
    return food_all


merged_df['aggregated_compound_embedding'] = merged_df.apply(concatenate_embeddings, axis=1)

print(merged_df.shape)

# Rename the final embedding column for clarity
food_embeddings = merged_df.rename(columns={'aggregated_compound_embedding': 'unified_embedding'})

# # # Step 4: Final DataFrame with food_name and unified embeddings
final_df = food_embeddings[['food_name', 'unified_embedding']]

# # # # Print or Save the Final Unified DataFrame
print(final_df.head)
print(final_df.shape)

final_df.to_pickle('C:/Users/labro/Downloads/Thesis_Food/final_unified_embeddings_aggregated_250.pkl')
