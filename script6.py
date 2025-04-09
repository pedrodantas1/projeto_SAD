import pandas as pd

# Carregar o CSV com coordenadas geográficas dos estados
coordenadas_estados = pd.read_csv('estados.csv')

# Verificar a estrutura do arquivo de coordenadas
print("Estrutura do arquivo de coordenadas:")
print(coordenadas_estados.head())
print("\nColunas disponíveis:", coordenadas_estados.columns.tolist())

# Carregar o arquivo de dados que queremos enriquecer com coordenadas
# Por exemplo, o arquivo de vulnerabilidade educacional
dados_vulnerabilidade = pd.read_csv('star_schema/agregacoes/indice_vulnerabilidade_educacional.csv')

# Verificar a estrutura do arquivo de dados
print("\nEstrutura do arquivo de dados:")
print(dados_vulnerabilidade.head())
print("\nColunas disponíveis:", dados_vulnerabilidade.columns.tolist())

# Realizar o merge dos dois dataframes
# Assumindo que ambos têm uma coluna 'uf' que pode ser usada como chave
dados_combinados = pd.merge(
    dados_vulnerabilidade,
    coordenadas_estados,
    on='uf',  # Coluna em comum para o merge
    how='left'  # Mantém todas as linhas do dataframe da esquerda
)

# Verificar o resultado do merge
print("\nEstrutura após o merge:")
print(dados_combinados.head())
print("\nColunas após o merge:", dados_combinados.columns.tolist())

# Remover colunas desnecessárias codigo_uf, nome, regiao
dados_combinados = dados_combinados.drop(columns=['codigo_uf', 'nome', 'regiao'])

# Verificar se há estados sem coordenadas (valores NaN)
estados_sem_coordenadas = dados_combinados[dados_combinados['latitude'].isna()]['uf'].unique()
if len(estados_sem_coordenadas) > 0:
    print("\nAtenção: Os seguintes estados não têm coordenadas:", estados_sem_coordenadas)

# Salvar o resultado em um novo arquivo CSV
dados_combinados.to_csv('star_schema/agregacoes/indice_vulnerabilidade_com_coordenadas.csv', index=False)
print("\nArquivo combinado salvo com sucesso!")

# Exemplo de como fazer o mesmo para o arquivo de concentração de analfabetismo por município
# Neste caso, precisaríamos de um arquivo com coordenadas de municípios
# try:
#     # Carregar dados de analfabetismo por município
#     dados_municipios = pd.read_csv('star_schema/agregacoes/concentracao_analfabetismo_municipios.csv')
    
#     # Carregar coordenadas de municípios (se disponível)
#     # coordenadas_municipios = pd.read_csv('dados/coordenadas_municipios.csv')
    
#     # Para este exemplo, vamos apenas mostrar como seria o merge
#     print("\nPara agregar dados de municípios, seria necessário um arquivo com coordenadas municipais.")
#     print("O processo seria semelhante, mas usando a combinação de 'nome_municipio' e 'uf' como chave.")
    
#     # O merge seria assim:
#     # dados_municipios_geo = pd.merge(
#     #     dados_municipios,
#     #     coordenadas_municipios,
#     #     on=['nome_municipio', 'uf'],
#     #     how='left'
#     # )
    
# except FileNotFoundError:
#     print("\nArquivo de concentração de analfabetismo por município não encontrado.")