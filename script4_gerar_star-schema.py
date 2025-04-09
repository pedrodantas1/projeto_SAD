import random
import pandas as pd
import numpy as np
import os

# Copiar base original
df = pd.read_csv("fato_alfabetizacao_com_siglas_uf.csv")

# Criar pasta para armazenar os arquivos do star schema
os.makedirs("star_schema", exist_ok=True)

# Tabela Dim_Tempo com ano e semestre ordenados sem indices
dim_tempo = df[["ano", "semestre"]].drop_duplicates().reset_index(drop=True)
dim_tempo = dim_tempo.sort_values(by=["ano", "semestre"]).reset_index(drop=True)
dim_tempo["id_tempo"] = dim_tempo.index + 1
dim_tempo = dim_tempo[["id_tempo"] + dim_tempo.columns[:-1].tolist()]

print("Dimensão Tempo:")
print(dim_tempo)
# Salvar dim_tempo
dim_tempo.to_csv("star_schema/dim_tempo.csv", index=False)

# Tabela Dim_Municipio
dim_municipio = df[["nome_municipio", "uf"]].drop_duplicates().reset_index(drop=True)

# Adicionar região com base na UF
regioes = {
    "AC": "Norte", "AL": "Nordeste", "AM": "Norte", "AP": "Norte", "BA": "Nordeste",
    "CE": "Nordeste", "DF": "Centro-Oeste", "ES": "Sudeste", "GO": "Centro-Oeste",
    "MA": "Nordeste", "MG": "Sudeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "PA": "Norte", "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "PR": "Sul",
    "RJ": "Sudeste", "RN": "Nordeste", "RO": "Norte", "RR": "Norte", "RS": "Sul",
    "SC": "Sul", "SE": "Nordeste", "SP": "Sudeste", "TO": "Norte"
}
dim_municipio["regiao"] = dim_municipio["uf"].map(regioes)
dim_municipio["id_municipio"] = dim_municipio.index + 1
dim_municipio = dim_municipio[["id_municipio"] + dim_municipio.columns[:-1].tolist()]
# Ordenar regiao, uf e nome do municipio
dim_municipio = dim_municipio.sort_values(by=["regiao", "uf", "nome_municipio"])

print("\nDimensão Município:")
print(dim_municipio.head())
# Salvar dim_municipio
dim_municipio.to_csv("star_schema/dim_municipio.csv", index=False)

# Tabela Dim_Demografica (combinando sexo, cor_raca e grupo_idade)
dim_demografica = df[["sexo", "cor_raca", "grupo_idade"]].drop_duplicates().reset_index(drop=True)
dim_demografica["id_demografica"] = dim_demografica.index + 1
dim_demografica = dim_demografica[["id_demografica"] + dim_demografica.columns[:-1].tolist()]

# Ordenar sexo, cor_raca e grupo_idade
dim_demografica = dim_demografica.sort_values(by=["sexo", "cor_raca", "grupo_idade"])

print("\nDimensão Demográfica:")
print(dim_demografica.head())
# Salvar dim_demografica
dim_demografica.to_csv("star_schema/dim_demografica.csv", index=False)

# Criar tabela fato com chaves estrangeiras
# Primeiro, criar mapeamentos de valores para IDs
tempo_map = dict(zip(dim_tempo[["ano", "semestre"]].apply(tuple, axis=1), dim_tempo["id_tempo"]))
municipio_map = dict(zip(dim_municipio[["nome_municipio", "uf"]].apply(tuple, axis=1), dim_municipio["id_municipio"]))
demografica_map = dict(zip(dim_demografica[["sexo", "cor_raca", "grupo_idade"]].apply(tuple, axis=1), dim_demografica["id_demografica"]))

# Criar a tabela fato
fato_alfabetizacao = df.copy()

# Adicionar chaves estrangeiras
fato_alfabetizacao["id_tempo"] = fato_alfabetizacao[["ano", "semestre"]].apply(tuple, axis=1).map(tempo_map)
fato_alfabetizacao["id_municipio"] = fato_alfabetizacao[["nome_municipio", "uf"]].apply(tuple, axis=1).map(municipio_map)
fato_alfabetizacao["id_demografica"] = fato_alfabetizacao[["sexo", "cor_raca", "grupo_idade"]].apply(tuple, axis=1).map(demografica_map)

# Remover colunas originais que agora são representadas por chaves estrangeiras
fato_alfabetizacao = fato_alfabetizacao.drop(columns=["ano", "semestre", "nome_municipio", "uf", "sexo", "cor_raca", "grupo_idade"])

# Adicionar um ID único para a tabela fato
fato_alfabetizacao["id_fato"] = range(0, len(fato_alfabetizacao))

# Reorganizar colunas para ter IDs no início
colunas = ["id_fato", "id_municipio", "id_tempo", "id_demografica"] + [
    col for col in fato_alfabetizacao.columns 
    if col not in ["id_fato", "id_municipio", "id_tempo", "id_demografica"]
]
fato_alfabetizacao = fato_alfabetizacao[colunas]

print("\nTabela Fato:")
print(fato_alfabetizacao.head())
# Salvar fato_alfabetizacao
fato_alfabetizacao.to_csv("star_schema/fato_alfabetizacao.csv", index=False)

print("\nArquivos CSV para o esquema estrela foram gerados na pasta 'star_schema':")
print("1. dim_tempo.csv")
print("2. dim_municipio.csv")
print("3. dim_demografica.csv")
print("4. fato_alfabetizacao.csv")