import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Carregar dados
correlacao_demografica = pd.read_csv('star_schema/agregacoes/correlacao_demografica_alfabetizacao.csv')

# Criar pasta para gráficos se não existir
os.makedirs("star_schema/graficos", exist_ok=True)

# Preparar dados para o heatmap (transpor para ter UFs nas linhas e fatores nas colunas)
corr_heatmap = correlacao_demografica.set_index('uf')

# Criar um heatmap para visualizar as correlações
plt.figure(figsize=(14, 10))
mask = np.zeros_like(corr_heatmap, dtype=bool)
heatmap = sns.heatmap(corr_heatmap, annot=True, cmap='coolwarm', center=0, 
                     vmin=-1, vmax=1, fmt='.2f', linewidths=.5)
plt.title('Correlação entre Fatores Demográficos e Taxa de Alfabetização por UF', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("star_schema/graficos/heatmap_correlacao_demografica.png", dpi=300, bbox_inches='tight')
plt.close()

# Criar gráficos de barras para cada fator demográfico
fatores = corr_heatmap.columns.tolist()
for fator in fatores:
    plt.figure(figsize=(12, 8))
    # Ordenar UFs pelo valor da correlação
    dados_ordenados = corr_heatmap.sort_values(by=fator, ascending=False)
    
    # Definir cores baseadas no valor (positivo=azul, negativo=vermelho)
    cores = ['red' if x < 0 else 'blue' for x in dados_ordenados[fator]]
    
    # Criar gráfico de barras
    plt.bar(dados_ordenados.index, dados_ordenados[fator], color=cores)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Correlação entre {fator} e Taxa de Alfabetização por UF', fontsize=14)
    plt.ylabel('Coeficiente de Correlação')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"star_schema/graficos/correlacao_{fator}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("Agregações semestrais criadas com sucesso:")
print("\nGráficos de correlação salvos em star_schema/graficos/")