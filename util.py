import pandas as pd

# Carregar o arquivo CSV
caminho_arquivo = r'fato_alfabetizacao_com_siglas_uf.csv'
dados = pd.read_csv(caminho_arquivo)

# Apagar colunas ano e semestre
dados.drop(['ano', 'semestre'], axis=1, inplace=True)

# Exibir as primeiras linhas do DataFrame
print(dados.head())

# Salvar o DataFrame modificado
dados.to_csv('Alfabetizacao_Data_Mining.csv', index=False)

print("As colunas 'ano' e 'semestre' foram apagadas e o DataFrame foi salvo.")