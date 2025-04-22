import pandas as pd
import numpy as np

df_foodb = pd.read_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/foodb/FoodDB.csv', sep=';')

print(df_foodb.shape)
# print(df_foodb.columns)

df_foodb = df_foodb[['id', 'name', 'Parent_SMILES']]
foodb_predictions = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/foodb/FooDB_predictions.txt', sep='\t', header=None)

print(foodb_predictions.shape)

foodb_predictions.drop(columns=[0,2,4,6], inplace=True)
print(foodb_predictions)
foodb_predictions_df = pd.DataFrame(columns=['bitter', 'sweet', 'umami', 'other']
                                    , data=foodb_predictions.values)

# print(foodb_predictions_df)

merged_df = pd.merge(df_foodb, foodb_predictions_df, left_index=True, right_index=True)
# print(merged_df)

bad_lines = []

merged_df_compound_predictions = merged_df[['name', 'bitter', 'sweet', 'umami', 'other']]
# merged_df.to_csv('C:/Users/labro/Downloads/FoodDB_compounds_and_predictions.csv', index=False)
# merged_df_compound_predictions.to_csv('C:/Users/labro/Downloads/Thesis_Food/FoodDB_compoundnames_and_predictions_only.csv', index=False, sep=';')
def bad_line_handler(line):
    bad_lines.append(line)
    return None
import csv
# food_compounds = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/foodb/foods_compounds.csv', sep=',', on_bad_lines=bad_line_handler, engine='python')
# food_compounds = pd.read_csv('C:/Users/labro/Downloads/Thesis_Food/foodb/foods_compounds.csv', sep=',')
# print(food_compounds.shape)
food_compounds = pd.read_csv('/Users/lamprosandroutsos/Documents/Thesis/Thesis_Food/foodb/foods_compounds_simplified.csv', sep=',')
# Handle bad lines with custom function
print(food_compounds.shape)
print('these many nan compound ids:', food_compounds['compound_id'].isnull().sum())

print("Total unique compounds in food_compounds:", food_compounds['compound_id'].nunique())
print("Total unique compounds in merged_df:", merged_df['id'].nunique())

# df = pd.read_csv('your_data_file.csv', on_bad_lines=bad_line_handler)

# Merging the dataframes to replace compound_id with the name from the compound table
merged_df_2 = food_compounds.merge(merged_df, left_on='compound_id', right_on='id', how='right')
print("Rows after merge:", len(merged_df_2))
print("Unique compounds after merge:", merged_df_2['compound_id'].nunique())

merged_df_2 = merged_df_2[['food_id', 'food_name', 'name']]
print("Rows after column selection:", len(merged_df_2))

# Detailed NaN analysis
print("\nNaN Analysis:")
print("Columns with NaN values:")
for col in merged_df_2.columns:
    nan_count = merged_df_2[col].isna().sum()
    if nan_count > 0:
        print(f"{col}: {nan_count} NaN values")

# Find compounds that are dropped due to NaN
dropped_compounds = merged_df_2[merged_df_2.isna().any(axis=1)]['name'].unique()
print("\nUnique compounds dropped due to NaN:", len(dropped_compounds))
print("Sample of dropped compound names:\n", dropped_compounds[:10])

print("\nExample Rows to be Dropped:")
rows_to_drop = merged_df_2[merged_df_2.isna().any(axis=1)]
print("Total rows to be dropped:", len(rows_to_drop))

if len(rows_to_drop) > 0:
    print("\nFirst few rows to be dropped:")
    print(rows_to_drop.head())
    
    print("\nNaN Analysis for First Dropped Row:")
    first_dropped_row = rows_to_drop.iloc[0]
    for col in merged_df_2.columns:
        if pd.isna(first_dropped_row[col]):
            print(f"{col}: NaN")
        else:
            print(f"{col}: {first_dropped_row[col]}")
            
# print(merged_df.shape)
merged_df_2.dropna(inplace=True)
print("Rows after dropping NaN:", len(merged_df_2))
print("Unique compounds after dropping NaN:", merged_df_2['name'].nunique())

missing_compounds = food_compounds[~food_compounds['compound_id'].isin(merged_df_2['name'])]
print("\nMissing compounds count:", len(missing_compounds))
print("Sample of missing compounds:\n", missing_compounds['compound_id'].head())


# Investigate merge mismatches
print("\nMerge Mismatch Diagnostics:")
# Check unique values in merge keys
print("Unique compound_id in food_compounds:", food_compounds['compound_id'].nunique())
print("Unique id in merged_df:", merged_df['id'].nunique())

# Find compounds in food_compounds that don't match in merged_df
unmatched_compounds = food_compounds[~food_compounds['compound_id'].isin(merged_df['id'])]
print("\nUnmatched compounds count:", len(unmatched_compounds))

# Sample of unmatched compounds
print("\nSample of unmatched compounds:")
print(unmatched_compounds[['compound_id']].head())

# Investigate the merge operation more closely
merged_diagnostic = food_compounds.merge(merged_df, left_on='compound_id', right_on='id', how='left', indicator=True)
print("\nMerge Diagnostic:")
print(merged_diagnostic['_merge'].value_counts())
# print(merged_df_2)
# print(merged_df_2['name'].nunique())
# print(merged_df['id'].nunique())



## 0-1 encoding of compound presence in foods ###
# Create a pivot table with food_name as rows, name as columns, and presence as values
pivot_table = merged_df_2.pivot_table(index='food_name', columns='name', aggfunc='size', fill_value=0)

# Convert the counts to 1 (if a compound is present) or 0 (if absent)
pivot_table = (pivot_table > 0).astype(int)

# Display the resulting DataFrame
print(pivot_table)
print(pivot_table.head)
print(pivot_table.columns)  
print(pivot_table.index)
print(pivot_table.shape)

# pivot_table.to_csv('C:/Users/labro/Downloads/Thesis_Food/foodname_compound_presence_0_1.csv', sep=';')


# merged_df.to_csv('C:/Users/labro/Downloads/food_compound_foodb.csv', index=False, sep=';')
merged_full_df = merged_df_2.merge(merged_df, left_on='name', right_on='name', how='left')

print(merged_full_df.shape)
# Grouping and formatting the data as requested
# grouped_compounds_df = merged_full_df.groupby('food_name').apply(
#     lambda x: list(zip(
#         x['name'],
#         zip(x['bitter'], x['sweet'], x['umami'], x['other'])
#     ))
# ).reset_index()
# grouped_compounds_df.columns = ['food_name', 'compounds']
grouped_compounds_df_bitter = merged_full_df.groupby('food_name').apply(
    lambda x: dict(zip(
        x['name'],
        x['bitter']
    ))
).reset_index()
grouped_compounds_df_bitter.columns = ['food_name', 'compounds']

print(grouped_compounds_df_bitter.shape)
grouped_compounds_df_sweet = merged_full_df.groupby('food_name').apply(
    lambda x: dict(zip(
        x['name'],
        x['sweet']
    ))
).reset_index()
grouped_compounds_df_sweet.columns = ['food_name', 'compounds']

grouped_compounds_df_umami = merged_full_df.groupby('food_name').apply(
    lambda x: dict(zip(
        x['name'],
        x['umami']
    ))
).reset_index()
grouped_compounds_df_umami.columns = ['food_name', 'compounds']

grouped_compounds_df_other = merged_full_df.groupby('food_name').apply(
    lambda x: dict(zip(
        x['name'],
        x['other']
    ))
).reset_index()
grouped_compounds_df_other.columns = ['food_name', 'compounds']

# grouped_compounds_df_bitter.to_csv('C:/Users/labro/Downloads/food_compound_foodb_grouped_bitter_dict.csv', index=False, sep=';')
# grouped_compounds_df_sweet.to_csv('C:/Users/labro/Downloads/food_compound_foodb_grouped_sweet_dict.csv', index=False, sep=';')
# grouped_compounds_df_umami.to_csv('C:/Users/labro/Downloads/food_compound_foodb_grouped_umami_dict.csv', index=False, sep=';')
# grouped_compounds_df_other.to_csv('C:/Users/labro/Downloads/food_compound_foodb_grouped_other_dict.csv', index=False, sep=';')

# Grouping the data by food_name and aggregating the compound_names into a list
# compounds_df = merged_df.groupby('food_name')['name'].apply(list).reset_index()

# # Renaming the columns as requested
# compounds_df.rename(columns={'compound_name': 'compounds'}, inplace=True)

# print(grouped_compounds_df)
# Log bad lines for review
# with open('C:/Users/labro/Downloads/bad_lines.log', 'w') as f:
#     for line in bad_lines:
#         f.write(f"{line}\n")