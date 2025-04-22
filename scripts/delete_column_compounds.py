import pandas as pd

df_bitter = pd.read_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/df_food_dict_bitter_embeddings_correct.csv', sep=';')
df_sweet = pd.read_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/df_food_dict_sweet_embeddings_correct.csv', sep=';')
df_umami = pd.read_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/df_food_dict_umami_embeddings_correct.csv', sep=';')
df_other = pd.read_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/df_food_dict_other_embeddings_correct.csv', sep=';')

df_bitter[['food_name', 'taste_embedding']].to_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/bitter_food_embeddings_correct.csv', index=False, sep=';')
df_sweet[['food_name', 'taste_embedding']].to_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/sweet_food_embeddings_correct.csv', index=False, sep=';')
df_umami[['food_name', 'taste_embedding']].to_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/umami_food_embeddings_correct.csv', index=False, sep=';')
df_other[['food_name', 'taste_embedding']].to_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/other_food_embeddings_correct.csv', index=False, sep=';')
