import numpy as np
# import tensorflow as tf
import pandas as pd
# from tensorflow.keras import layers, models, backend as K
import torch
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import umap
import pickle

# final_df = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/final_unified_embeddings.pkl')
final_df = pd.read_pickle('C:/Users/labro/Downloads/Thesis_Food/embeddings/df_filtered_food_50.pkl')
# # print(final_df)
embeddigns_list = final_df['unified_embedding'].tolist()

# print first value of unified_embedding, first array




# embeddings_array = np.vstack(embeddigns_list)
print(embeddings_array.shape)

# t = torch.from_numpy(embeddings_array)

# print(t.shape)
from sklearn.decomposition import PCA
# Step 2: Apply PCA to reduce dimensions (e.g., from 400 to 50)
pca_model = PCA(n_components=30)
# pca_reduced_embeddings = pca_model.fit(embeddings_array)

# # print('PCA reduced embeddings shape:', pca_reduced_embeddings.shape)
# # Calculate cumulative explained variance
# cumulative_variance = np.cumsum(pca_reduced_embeddings.explained_variance_ratio_)
# # Check sum of explained variance ratio to confirm it sums to ~1
# total_explained_variance = np.sum(pca_reduced_embeddings.explained_variance_ratio_)
# print(f"Total exp3lained variance: {total_explained_variance}")
# # Find the number of components needed to reach 90% variance
# n_components_90_var = np.argmax(cumulative_variance >= 0.90) + 1
# print(f"Number of components for 90% variance: {cumulative_variance}")

pca_reduced_embeddings = pca_model.fit_transform(embeddings_array)

print(f"Shape of PCA reduced embeddings: {pca_reduced_embeddings.shape}")  # (num_foods, 50)

# save the pca embeddings to pickle file. Dont add it to the dataframe.
with open("C:/Users/labro/Downloads/Thesis_Food/pca_embeddings_30.pkl", "wb") as f:
    pickle.dump(pca_reduced_embeddings, f)

print('pickle dump pca done')
# Step 2: Apply UMAP for dimensionality reduction
# Here we reduce the embeddings to 50 dimensions (you can adjust `n_components` as needed)
umap_model = umap.UMAP(n_components=10, random_state=42, low_memory=True, n_jobs=10, verbose=True)

# Fit UMAP on the flattened embeddings
reduced_embeddings = umap_model.fit_transform(pca_reduced_embeddings)

print(f"Shape of reduced embeddings: {reduced_embeddings.shape}")  # (num_foods, 50)

# save the umap embeddings to pickle file
with open("C:/Users/labro/Downloads/Thesis_Food/umap_embeddings_30pca_10.pkl", "wb") as f:
    pickle.dump(reduced_embeddings, f)


# Step 3: Reshape the reduced embeddings back into foods, and aggregate them by food
# Assuming you have 400-dim embeddings for each compound, and you have different lengths of compound lists per food
food_lengths = embeddings_array['unified_embedding'].apply(len).values  # Get the length of embeddings per food

# Split the reduced embeddings back into a list of arrays, one per food
split_embeddings = np.split(reduced_embeddings, np.cumsum(food_lengths)[:-1])

# Optionally, you can mean-pool the reduced embeddings across all compounds for each food:
mean_pooled_embeddings = np.array([np.mean(food, axis=0) for food in split_embeddings])

# Now you have `mean_pooled_embeddings` with a fixed size per food
print(f"Shape of mean-pooled embeddings: {mean_pooled_embeddings.shape}")  # (num_foods, 50)