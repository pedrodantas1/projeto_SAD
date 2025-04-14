import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *

# Carregando dados já processados
data = pd.read_csv('municipios_agregados.csv')

# Setup do experimento
exp = setup(
    data=data,
    target='vulnerabilidade_educacional',
    numeric_features=['taxa_alfabetizacao'],
    categorical_features=[col for col in data.columns if col.startswith('prop_')],
    ignore_features=['nome_municipio', 'uf', 'pop_alfabetizada', 'populacao_total', 'indice_complexo'],
    transformation=True,
    normalize=True,
    fix_imbalance=True,
    fold=10,
    session_id=42
)

# Criando e comparando modelos
print("\n=== Criando e comparando modelos ===")
models = compare_models(include=['dt', 'rf'], 
                      fold=10, 
                      sort='Accuracy', 
                      n_select=2,
                      verbose=True)

# Obtendo métricas
metrics = pull()

# Criando visualização
plt.figure(figsize=(15, 6))

# Subplot 1: Métricas gerais
plt.subplot(1, 2, 1)
metrics_to_plot = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']
ax = metrics[metrics_to_plot].plot(kind='bar', ax=plt.gca())

# Adicionando labels nas barras
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)

plt.title('Comparação de Métricas')
plt.xticks(rotation=45)
plt.ylim(0, 1.1)  # Aumentando um pouco o limite Y para as labels não ficarem cortadas

# Subplot 2: Tempo de treino e predição
# plt.subplot(1, 2, 2)
# ax2 = metrics[['TT (Sec)', 'PT (Sec)']].plot(kind='bar', ax=plt.gca())

# # Adicionando labels nas barras do segundo gráfico
# for container in ax2.containers:
#     ax2.bar_label(container, fmt='%.3f', padding=3)

# plt.title('Comparação de Tempo (segundos)')
# plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Dados_PowerBI/comparacao_modelos.png', dpi=300, bbox_inches='tight')

# Salvando métricas detalhadas
metrics.round(4).to_csv('Dados_PowerBI/metricas_modelos.csv', encoding='utf-8-sig')

print("\nComparação dos modelos:")
print(metrics.round(4))
print("\nVisualizações e métricas salvas em 'Dados_PowerBI/'")