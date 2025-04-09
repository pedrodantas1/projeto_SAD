import pandas as pd

# Carregar o arquivo CSV
caminho_arquivo = r'star_schema/agregacoes/indice_vulnerabilidade_educacional.csv'
dados = pd.read_csv(caminho_arquivo)

# Calcular a média da coluna taxa_analfabetismo
media_taxa = dados['taxa_analfabetismo'].mean()

# Exibir o resultado
print(f"A média da taxa de analfabetismo é: {media_taxa:.4f}%")

# Se quiser calcular a média de outras colunas também
media_indice = dados['indice_vulnerabilidade'].mean()
print(f"A média do índice de vulnerabilidade é: {media_indice:.4f}")