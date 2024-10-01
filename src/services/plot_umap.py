import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

pd.set_option('display.max_columns', None)
df = pd.read_csv('./llama-umap/data/Natural-Questions-LLM-responses-400.csv')

print(df.shape)
print(df.columns)

def get_embedding(input_str):
    input_str = input_str[2:]
    input_str = input_str[:-2]
    input_str = list(input_str.split(", "))
    float_lst = [float(item) for item in input_str]
    res = float_lst
    return res

df['ta_embd_llama3'] = df['true_answer_embedding_llama3.1'].apply(lambda x: get_embedding(x))
df['ta_embd_gemma2'] = df['true_answer_embedding_gemma2'].apply(lambda x: get_embedding(x))
df['ta_embd_phi3'] = df['true_answer_embedding_phi3'].apply(lambda x: get_embedding(x))

df = df.drop(columns=['true_answer_embedding_gemma2', 'true_answer_embedding_llama3.1', 'true_answer_embedding_phi3'])

df_llama = df[['ta_embd_llama3']]
df_gemma = df[['ta_embd_gemma2']]
df_phi3 = df[['ta_embd_phi3']]
print('LLM dataset:')
print(df_llama.columns)
print(df_llama.shape)

# 1. Convert list of floats to multiple columns of floats for each dataframe
def expand_embedding_column(df, col_name):
    return pd.DataFrame(df[col_name].tolist(), index=df.index)

# Expand embeddings into separate columns
df_llama_expanded = expand_embedding_column(df_llama, 'ta_embd_llama3')
df_gemma_expanded = expand_embedding_column(df_gemma, 'ta_embd_gemma2')
df_phi3_expanded = expand_embedding_column(df_phi3, 'ta_embd_phi3')

print(f'Llama dataset     : {df_llama_expanded.shape}')
print(f'Gemma dataset     : {df_gemma_expanded.shape}')
print(f'Phi3  dataset     : {df_phi3_expanded.shape}')

# 2. Pad embeddings to ensure they have the same length
def pad_embeddings(df, max_len):
    return pd.DataFrame(np.array([np.pad(embedding, (0, max_len - len(embedding)), 'constant') 
                                  for embedding in df.values]))

# Find the maximum embedding length across all models
max_len = max(df_llama_expanded.shape[1], df_gemma_expanded.shape[1], df_phi3_expanded.shape[1])

# Pad the embeddings to ensure uniform length
# (The most controversial step. Maybe, we should trunk vectors, not pad)
df_llama_padded = pad_embeddings(df_llama_expanded, max_len)
df_gemma_padded = pad_embeddings(df_gemma_expanded, max_len)
df_phi3_padded = pad_embeddings(df_phi3_expanded, max_len)

# 3. Merge the dataframes, adding a label column for the model (0 for llama, 1 for gemma, 2 for phi3)
df_llama_padded['label'] = 0  # Llama model
df_gemma_padded['label'] = 1  # Gemma model
df_phi3_padded['label'] = 2   # Phi3 model

# Combine all padded dataframes
df_combined = pd.concat([df_llama_padded, df_gemma_padded, df_phi3_padded], ignore_index=True)

print('---------------------------------')
print(f'Combined  dataset : {df_combined.shape}')

gemma_row = df_combined.iloc[1160]

#print(gemma_row)

# Separate embeddings (all columns except 'label') and labels
embeddings = df_combined.drop(columns=['label']).values  # embeddings data
labels = df_combined['label'].values  # model labels

# 3. Apply UMAP
umap_model = umap.UMAP(n_neighbors=8, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = umap_model.fit_transform(embeddings)

# 4. Plot the UMAP projection
plt.figure(figsize=(10, 8))

# Scatter plot for each model with different colors
plt.scatter(embedding_2d[labels == 0, 0], embedding_2d[labels == 0, 1], label="Llama3", alpha=0.7, s=50, c='r')
plt.scatter(embedding_2d[labels == 1, 0], embedding_2d[labels == 1, 1], label="Gemma2", alpha=0.7, s=50, c='g')
plt.scatter(embedding_2d[labels == 2, 0], embedding_2d[labels == 2, 1], label="Phi3", alpha=0.7, s=50, c='b')

# Add titles and labels
plt.title("UMAP Projection of Embeddings from Llama, Gemma, and Phi Models", fontsize=16)
plt.xlabel("UMAP 1", fontsize=12)
plt.ylabel("UMAP 2", fontsize=12)

# Add legend
plt.legend()

# Show the plot
plt.savefig('./pics/umap_5.png')
plt.show()
