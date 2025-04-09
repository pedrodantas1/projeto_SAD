import pandas as pd

# Load your main data file
df_main = pd.read_csv("fato_alfabetizacao_pre_pronto_v2.csv")

# Load the municipality reference table that contains id_municipio and nome_municipio
df_municipios = pd.read_csv("br_bd_diretorios_brasil_municipio.csv")  # Adjust filename as needed

# Print the first few rows of each dataframe to verify the structure
# print("Main dataframe:")
# print(df_main[['id_municipio']].head())
# print("\nMunicipality reference dataframe:")
# print(df_municipios.head())

# Merge the dataframes to add the municipality names
# Assuming the column in df_municipios is called 'id' and 'nome'
df_merged = pd.merge(
    df_main,
    df_municipios[['id_municipio', 'nome', 'nome_uf']],
    left_on='id_municipio',
    right_on='id_municipio',
    how='left'
)

# Replace id_municipio with nome_municipio
df_merged['id_municipio_original'] = df_merged['id_municipio']
df_merged['id_municipio'] = df_merged['nome']


# Rename the new column to 'nome_municipio'
df_merged.columns = df_merged.columns.str.replace('nome', 'nome_municipio')

# Drop the redundant columns
df_merged = df_merged.drop(['id_municipio', 'id_municipio_original'], axis=1)

# Rename the new column to 'nome_municipio'
df_merged.columns = df_merged.columns.str.replace('nome_municipio_uf', 'nome_uf')

# Colocar nome_municipio como primeira coluna e nome_uf como segunda
cols = list(df_merged.columns)
nome_municipio_index = cols.index('nome_municipio')
nome_uf_index = cols.index('nome_uf')
cols.pop(nome_municipio_index)
cols.pop(nome_uf_index - 1 if nome_uf_index > nome_municipio_index else nome_uf_index)
cols = ['nome_municipio', 'nome_uf'] + cols

df_merged = df_merged[cols]

print(df_merged.head())


# Verificar se linhas duplicadas e printar quantidade de duplicações e printar linhas duplicadas
print("\nDuplicated rows:")
print(df_merged.duplicated().sum())

# Printar linhas duplicadas
print(df_merged[df_merged.duplicated()])

# Remover linhas duplicadas
df_merged = df_merged.drop_duplicates()

# Verificar se linhas duplicadas e printar quantidade de duplicações e printar linhas duplicadas
print("\nDuplicated rows after:")
print(df_merged.duplicated().sum())

# Verificar se há linhas com valores faltantes
print("\nMissing values:")
print(df_merged.isnull().sum())

# Remover linhas com valores faltantes
df_merged = df_merged.dropna()

# Save the updated dataframe
df_merged.to_csv("fato_alfabetizacao_com_municipios.csv", index=False)