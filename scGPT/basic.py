import pandas as pd

df = pd.read_parquet("data/cellxgene/batch_download_scb/blood/partition_0.scb/counts.datatable.parquet")
print("hello")
print(df.head())
print(df.columns)

# Add this to basic.py to debug
print("Expression list lengths:")
print(df['expressions'].apply(len).value_counts())
print("\nFirst few expression lists:")
for i in range(3):
    print(f"Row {i} length: {len(df['expressions'].iloc[i])}")

print("\nGenes list lengths:")
print(df['genes'].apply(len).value_counts())
print("\nFirst few genes lists:")
for i in range(3):
    print(f"Row {i} length: {len(df['genes'].iloc[i])}")
    print(f"First few genes indices: {df['genes'].iloc[i][:5]}")