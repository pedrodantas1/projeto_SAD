import random
import pandas as pd
import numpy as np
import os

# Copiar base original
df = pd.read_csv("fato_alfabetizacao_com_municipios.csv")

# Criar pasta para armazenar os arquivos do star schema
os.makedirs("star_schema", exist_ok=True)

# Tabela Dim_Tempo com ano e semestre ordenados sem indices
dim_tempo = df[["ano", "semestre"]].drop_duplicates().reset_index(drop=True)
dim_tempo = dim_tempo.sort_values(by=["ano", "semestre"]).reset_index(drop=True)
dim_tempo["id_tempo"] = dim_tempo.index + 1

print("Dimensão Tempo:")
print(dim_tempo)
# Salvar dim_tempo
dim_tempo.to_csv("star_schema/dim_tempo.csv", index=False)

# Tabela Dim_Municipio
dim_municipio = df[["nome_municipio", "nome_uf"]].drop_duplicates().reset_index(drop=True)
dim_municipio["id_municipio"] = dim_municipio.index + 1

# Adicionar região com base na UF
regioes = {
    "AC": "Norte", "AL": "Nordeste", "AM": "Norte", "AP": "Norte", "BA": "Nordeste",
    "CE": "Nordeste", "DF": "Centro-Oeste", "ES": "Sudeste", "GO": "Centro-Oeste",
    "MA": "Nordeste", "MG": "Sudeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "PA": "Norte", "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "PR": "Sul",
    "RJ": "Sudeste", "RN": "Nordeste", "RO": "Norte", "RR": "Norte", "RS": "Sul",
    "SC": "Sul", "SE": "Nordeste", "SP": "Sudeste", "TO": "Norte"
}
dim_municipio["regiao"] = dim_municipio["nome_uf"].map(regioes)