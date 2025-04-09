import random
import pandas as pd
import numpy as np

# Copiar base original
df = pd.read_csv("fato_alfabetizacao_pre_pronto_v2.csv")

# Tabela Dim_Tempo com ano e semestre ordenados sem indices
dim_tempo = df[["ano", "semestre"]].drop_duplicates().reset_index(drop=True)
dim_tempo = dim_tempo.sort_values(by=["ano", "semestre"]).reset_index(drop=True)
dim_tempo["id_tempo"] = dim_tempo.index + 1

print(dim_tempo)

# Tabela Dim_Sexo
dim_sexo = df[["sexo"]].drop_duplicates().reset_index(drop=True)
dim_sexo["id_sexo"] = dim_sexo.index + 1

print(dim_sexo)

# Tabela Dim_Cor
dim_cor = df[["cor_raca"]].drop_duplicates().reset_index(drop=True)
dim_cor["id_cor"] = dim_cor.index + 1

print(dim_cor)

# Tabela Dim_Faixa_Etaria
dim_faixa = df[["grupo_idade"]].drop_duplicates().reset_index(drop=True)
dim_faixa = dim_faixa.sort_values(by=["grupo_idade"]).reset_index(drop=True)
dim_faixa["id_faixa_etaria"] = dim_faixa.index + 1

print(dim_faixa)

# Printar tipo das colunas
print(df.dtypes)

# Tabela Dim_Municipio (fazer ainda a correlacao)
municipios_unicos = df["id_municipio"].drop_duplicates().reset_index(drop=True)
dim_municipio = pd.DataFrame({
    "id_municipio": municipios_unicos,
    "nome_municipio": ["Municipio_" + str(i) for i in range(1, len(municipios_unicos)+1)],
    "uf": np.random.choice(["MG", "SP", "BA", "RS", "AM", "PR", "PE"], size=len(municipios_unicos))
})
# Adicionando região com base no estado (fazer tbm a correlação)
regioes = {
    "MG": "Sudeste", "SP": "Sudeste", "BA": "Nordeste", "RS": "Sul",
    "AM": "Norte", "PR": "Sul", "PE": "Nordeste"
}
dim_municipio["regiao"] = dim_municipio["uf"].map(regioes)
