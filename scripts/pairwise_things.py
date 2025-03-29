import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# df = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/FoodDB_compoundnames_and_predictions_only.csv', sep=';')
# print(df)

# df_bitter = df[['name', 'bitter']]
# df_sweet = df[['name', 'sweet']]
# df_umami = df[['name', 'umami']]
# df_other = df[['name', 'other']]

# df_bitter.to_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_bitter.csv', index=False, sep=';')
# df_sweet.to_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_sweet.csv', index=False, sep=';')
# df_umami.to_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_umami.csv', index=False, sep=';')
# df_other.to_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_other.csv', index=False, sep=';')

# df = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_other.csv', sep=';')
# # Step 2: Sort the compounds by prediction scores to determine the global order
# global_order_df = df.sort_values(by='other', ascending=False)['name']
# # global_order_df.to_csv('C:/Users/labro/Downloads/Thesis_Food/compounds_bitter_global_order.csv', index=False, sep=';')

# global_order = global_order_df.tolist()

# Check the global order of compounds
# print("Global Order of Compounds:")
# print(global_order)

# Load the CSV file
df = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/food_compound_foodb_grouped_other_dict.csv', sep=';')

print(df)
# count keys in the first row at compounds column
print(len(ast.literal_eval(df['compounds'][0])))

# Parse the compounds column
# df['compounds'] = df['compounds'].apply(ast.literal_eval)

# # Expand the compounds column into multiple rows
# df_expanded = df.explode('compounds')

# # # print(df_expanded.head())
# # # Split the tuple into compound and prediction score
# df_expanded[['compound', 'prediction_score']] = pd.DataFrame(df_expanded['compounds'].tolist(), index=df_expanded.index)
# df_expanded['prediction_score'] = df_expanded['prediction_score'].apply(lambda x: x[0])

# print(df_expanded.head())


# # # Drop the original compounds column
# # df_expanded = df_expanded.drop(columns=['compounds'])
# # df_expanded_duplicates = df_expanded.index[df_expanded.index.duplicated()]
# # print(f'Duplicates: {df_expanded_duplicates}')

# df_expanded.reset_index(drop=True, inplace=True)
# print(df_expanded.head())

# # Pivot the table to create a matrix with compounds as rows and food_name as columns
# # compound_matrix = df_expanded.pivot(index='compound', columns='food_name', values='prediction_score').fillna(0)
# compound_matrix = df_expanded.pivot_table(index='compound', columns='food_name', values='prediction_score').fillna(0)

# print(compound_matrix.head())

# Step 7: Collect the ordered compounds for each food in a dictionary
# ordered_dict = {}

# # Step 8: Filter and order the compounds for each food based on the global order
# for food in df_expanded['food_name'].unique():
#     print(food)
#     food_compounds = df_expanded[df_expanded['food_name'] == food]['compound'].tolist()
#     ordered_compounds = [comp for comp in global_order if comp in food_compounds]
#     ordered_dict[food] = ordered_compounds

# ordered_compounds_per_food = pd.concat({k: pd.Series(v) for k, v in ordered_dict.items()}, axis=1)

# # Step 8: Convert the dictionary to a DataFrame using pd.concat
# print("Ordered Compounds for Each Food:")
# print(ordered_compounds_per_food)
# Example: Sorting the compounds within each food by prediction score
# Ensure that 'compounds' is actually a dictionary
# Step 2: Identify and print entries that are strings instead of dictionaries
# for i, entry in enumerate(df['compounds']):
#     if isinstance(entry, str):
#         print(f"Entry at index {i} is a string: {entry}")

df['compounds'] = df['compounds'].apply(lambda x: x if isinstance(x, dict) else ast.literal_eval(x))


# df['sorted_compounds'] = df['compounds'].apply(
#     lambda x: dict(sorted(x.items(), key=lambda item: float(item[1]), reverse=True))
# )

# Step 3: Sorting the compounds within each food by prediction score, ensuring values are floats
# But only keeping the compound names
df['sorted_compounds'] = df['compounds'].apply(
    lambda x: [comp for comp, score in sorted(x.items(), key=lambda item: float(item[1]), reverse=True)]
)
print("DataFrame with Sorted Compounds by Prediction Score:")
# print(df[['food_name', 'sorted_compounds']])

# the length of the first sorted_compounds row
print(len(df['sorted_compounds'][0]))

# Optional: Save the ordered compounds for each food to a CSV file if needed
# df[['food_name', 'sorted_compounds']].to_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds_per_food_other.csv', index=False, sep=';')

# df_test = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/ordered_compounds_per_food_other.csv', sep=';')
# print(df_test)