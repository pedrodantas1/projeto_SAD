import pandas as pd

# Carregar o dataframe
df = pd.read_csv("fato_alfabetizacao_com_municipios.csv")

# Mapeamento de nomes completos para siglas das UFs do Brasil
mapeamento_ufs = {
    "Acre": "AC",
    "Alagoas": "AL",
    "Amapá": "AP",
    "Amazonas": "AM",
    "Bahia": "BA",
    "Ceará": "CE",
    "Distrito Federal": "DF",
    "Espírito Santo": "ES",
    "Goiás": "GO",
    "Maranhão": "MA",
    "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS",
    "Minas Gerais": "MG",
    "Pará": "PA",
    "Paraíba": "PB",
    "Paraná": "PR",
    "Pernambuco": "PE",
    "Piauí": "PI",
    "Rio de Janeiro": "RJ",
    "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS",
    "Rondônia": "RO",
    "Roraima": "RR",
    "Santa Catarina": "SC",
    "São Paulo": "SP",
    "Sergipe": "SE",
    "Tocantins": "TO"
}

# Verificar se a coluna nome_uf existe no dataframe
if "nome_uf" in df.columns:
    # Substituir o nome completo pela sigla
    df["uf"] = df["nome_uf"].map(mapeamento_ufs)
    
    # Opcional: remover a coluna original com o nome completo
    df.drop(columns=["nome_uf"], inplace=True)
    
    # print("Conversão concluída. Primeiras linhas do dataframe:")
    # print(df.head())
else:
    print("A coluna 'nome_uf' não foi encontrada no dataframe.")

# Verificar se houve linhas cujo valor do nome_uf nao foi convertido para a sigla
linhas_sem_conversao = df[df["uf"].isnull()]
if not linhas_sem_conversao.empty:
    print("As seguintes linhas não foram convertidas para a sigla:")
    print(linhas_sem_conversao[["nome_uf"]])
else:
    print("Todas as linhas foram convertidas para a sigla.")

# Colocar a coluna "uf" na segunda posição
colunas = df.columns.tolist()
colunas.remove("uf")
colunas.insert(1, "uf")
df = df[colunas]

print(df.head())

# Salvar o dataframe atualizado
df.to_csv("fato_alfabetizacao_com_siglas_uf.csv", index=False)