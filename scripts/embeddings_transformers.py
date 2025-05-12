import pandas as pd
import numpy as np
import ast
# from transformers import AutoTokenizer, AutoModel # Remove old imports
from sentence_transformers import SentenceTransformer # Add new import
import torch
from tqdm import tqdm

# Use relative paths
df_food_dict_bitter = pd.read_csv('ordered_compounds/ordered_compounds_per_food_bitter.csv', sep=';')
df_food_dict_sweet = pd.read_csv('ordered_compounds/ordered_compounds_per_food_sweet.csv', sep=';')
df_food_dict_umami = pd.read_csv('ordered_compounds/ordered_compounds_per_food_umami.csv', sep=';')
df_food_dict_other = pd.read_csv('ordered_compounds/ordered_compounds_per_food_other.csv', sep=';')

df_food_dict_bitter['sorted_compounds'] = df_food_dict_bitter['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df_food_dict_sweet['sorted_compounds'] = df_food_dict_sweet['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df_food_dict_umami['sorted_compounds'] = df_food_dict_umami['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df_food_dict_other['sorted_compounds'] = df_food_dict_other['sorted_compounds'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Merge compounds in order (bitter -> sweet -> umami -> other)
all_foods_compounds = []
all_tastes = []

# Concatenate compounds and keep track of tastes
for df, taste in [(df_food_dict_bitter, 'bitter'), 
                  (df_food_dict_sweet, 'sweet'),
                  (df_food_dict_umami, 'umami'),
                  (df_food_dict_other, 'other')]:
    compounds_list = df['sorted_compounds'].tolist()
    all_foods_compounds.extend(compounds_list)
    all_tastes.extend([taste] * len(compounds_list))

# Convert compounds lists to strings for the transformer
compound_texts = [' '.join(compounds) for compounds in all_foods_compounds]

# Initialize Sentence Transformer model
model_name = 'all-MiniLM-L6-v2' # Use sentence-transformer model
# tokenizer = AutoTokenizer.from_pretrained(model_name) # Remove old tokenizer
# model = AutoModel.from_pretrained(model_name) # Remove old model loading
model = SentenceTransformer(model_name) # Initialize SentenceTransformer

# Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # SentenceTransformer handles device placement internally by default
# model = model.to(device) # Not needed for default usage

# Generate embeddings
embeddings = []
# batch_size = 128 # SentenceTransformer's encode handles batching

# # Process in batches with tqdm progress bar # Old loop removed
# for i in tqdm(range(0, len(compound_texts), batch_size)):
#     batch_texts = compound_texts[i:i + batch_size]
#
#     # Tokenize and prepare input
#     inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     # Generate embeddings
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Use [CLS] token embedding as sequence representation
#         batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#
#     embeddings.extend(batch_embeddings)

# Use model.encode for simpler embedding generation (handles batching, device placement, progress bar)
embeddings = model.encode(compound_texts, show_progress_bar=True)


# Convert to numpy array # Already a numpy array from encode()
# embeddings = np.array(embeddings) # Not needed

# Create DataFrame with embeddings and tastes
embeddings_df = pd.DataFrame({
    'taste': all_tastes,
    'compounds': all_foods_compounds,
    'embedding': embeddings.tolist() # Keep as list for pickle compatibility if needed
})

# Save embeddings using relative paths - update filename for new model
embeddings_df.to_pickle('embeddings_data/food_compounds_embeddings_MiniLM_separate_tastes.pkl')

# Optional: Also save as numpy array - update filename for new model
np.save('embeddings_data/food_compounds_embeddings_MiniLM_separate_tastes.npy', embeddings)
