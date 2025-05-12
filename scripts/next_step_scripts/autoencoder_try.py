import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, backend as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

final_df = pd.read_pickle('/home/lamprosandroutsos/Documents/Thesis/Thesis_Food/embeddings/df_filtered_food_50.pkl')
# print(final_df)


## προσθετω τα διανυσματα των compounds μεταξυ τους 


# Create a subset with 10% of the data for faster testing
subset_df, _ = train_test_split(final_df, test_size=0.9, random_state=42)

# Prepare the subset embeddings array
subset_embeddings = np.vstack(subset_df['unified_embedding'].tolist()).astype(np.float32)

# Define the input embedding dimension and latent dimension
input_dim = 200
latent_dim = 64  # Bottleneck dimension

## Encoder (first try)
# inputs = layers.Input(shape=(input_dim,))
# encoded = layers.Dense(128, activation='relu')(inputs)  # First reduction layer
# encoded = layers.Dense(64, activation='relu')(encoded)  # Further reduction
# latent = layers.Dense(latent_dim, activation='relu')(encoded)  # Bottleneck layer

# # Decoder
# decoded = layers.Dense(64, activation='relu')(latent)
# decoded = layers.Dense(128, activation='relu')(decoded)
# outputs = layers.Dense(input_dim, activation='linear')(decoded)  # Output layer
# output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

# # Encoder (second try )
# inputs = layers.Input(shape=(input_dim,))
# x = layers.Dense(128, activation='relu')(inputs)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(64, activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)
# latent = layers.Dense(latent_dim, activation='relu')(x)

# # Decoder
# x = layers.Dense(64, activation='relu')(latent)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.BatchNormalization()(x)
# outputs = layers.Dense(input_dim, activation='linear')(x)


## Encoder (third try)
# Encoder
inputs = layers.Input(shape=(input_dim,))
x = layers.Dense(256, activation='relu')(inputs)  # Larger first layer
x = layers.Dense(128, activation='relu')(x)  # Compression step
latent = layers.Dense(128, activation='relu')(x)  # Latent space with 128 dimensions

# Decoder
x = layers.Dense(128, activation='relu')(latent)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(200, activation='linear')(x)  # Reconstruction layer

def combined_loss(y_true, y_pred):
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.5 * mae_loss + 0.5 * mse_loss
# from tensorflow.keras.losses import CosineSimilarity

# cosine_loss = CosineSimilarity(axis=-1)

# def adjusted_combined_loss(y_true, y_pred):
#     mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
#     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
#     cosine_loss_value = -cosine_loss(y_true, y_pred)
#     return 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * cosine_loss_value
# ## Encoder (Vae model)
# inputs = layers.Input(shape=(input_dim,))
# x = layers.Dense(128, activation='relu')(inputs)
# x = layers.Dense(64, activation='relu')(x)

# # Latent variables (mean and variance)
# z_mean = layers.Dense(latent_dim)(x)
# z_log_var = layers.Dense(latent_dim)(x)

# # Sampling layer
# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# z = layers.Lambda(sampling)([z_mean, z_log_var])

# # Decoder
# decoder_h1 = layers.Dense(64, activation='relu')
# decoder_h2 = layers.Dense(128, activation='relu')
# decoder_out = layers.Dense(input_dim, activation='linear')

# h1_decoded = decoder_h1(z)
# h2_decoded = decoder_h2(h1_decoded)
# outputs = decoder_out(h2_decoded)

# # Define VAE model
# vae = models.Model(inputs, outputs)

# # Custom VAE loss (reconstruction loss + KL divergence)
# from tensorflow.keras.losses import mse

vae = models.Model(inputs, outputs)

# reconstruction_loss = mse(inputs, outputs)

# kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
# vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

# # VAE model

# vae.add_loss(vae_loss)

# Compile VAE
from tensorflow.keras.optimizers import Adam

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=combined_loss)
# vae.compile(optimizer='adam')
embeddigns_list = final_df['unified_embedding'].tolist()
# # print(embeddigns_list)

embeddings_array = np.vstack(embeddigns_list).astype(np.float32)
# print(embeddings_array.shape)

# validation_split = 0.2  # Define the fraction of data to be used for validation
# total_size = len(final_df)  # Or the total number of items in your dataset
# val_size = int(validation_split * total_size)


def data_generator(embeddings, batch_size):
    data_size = len(embeddings)
    while True:
        for start in range(0, data_size, batch_size):
            end = min(start + batch_size, data_size)
            batch_data = embeddings[start:end]
            yield batch_data, batch_data  # Since it's an autoencoder

# Set batch size
batch_size = 5000

# # Create a tf.data.Dataset using the generator -> this is to lower memory usage
# dataset = tf.data.Dataset.from_generator(
#     lambda: data_generator(embeddings_array, batch_size),
#     output_signature=(
#         tf.TensorSpec(shape=(None, embeddings_array.shape[1]), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, embeddings_array.shape[1]), dtype=tf.float32)
#     )
# )

# dataset = tf.data.Dataset.from_tensor_slices((embeddings_array, embeddings_array))
# dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# # Prefetch for performance optimization
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# train_gen = data_generator(embeddings_array, batch_size)
# print('shape' ,embeddings_array[:600].shape)
# print('shape' ,embeddings_array.shape)

# Train the VAE (adjust epochs and batch_size based on your data)
# Reshape the input data to (batch_size, 1, embedding_dimension)
# embeddings_array_reshaped = np.expand_dims(embeddings_array, axis=1)  # Add a sequence length of 1

# Fit the VAE model with the reshaped input
# vae.fit(embeddings_array_reshaped, embeddings_array_reshaped, epochs=50, verbose=1)
with tf.device('/cpu:0'):
    history = vae.fit(embeddings_array, embeddings_array, epochs=20,  batch_size=1024,verbose=1)

## this is to lower memory usage
# vae.fit(dataset, epochs=100, steps_per_epoch=embeddings_array.shape[0] // batch_size,verbose=1)

history = vae.fit(embeddings_array, embeddings_array, epochs=20,  batch_size=256,verbose=1)
# history = vae.fit(subset_embeddings, subset_embeddings, epochs=20,  batch_size=256,verbose=1)


# Extract training loss for evaluation
training_loss = history.history['loss']
print(f"Final training loss: {training_loss[-1]}")
# vae.fit(dataset, epochs=50, verbose=1)

# Encoder model to get the latent representation
encoder = models.Model(inputs, outputs)
X_latent = encoder.predict(embeddings_array[:600])
df_latent = pd.DataFrame(X_latent)
df_latent.to_csv('../X_latent_dataframe_fulldata.csv', index=False)
print("Latent representations saved as 'X_latent.csv'")
