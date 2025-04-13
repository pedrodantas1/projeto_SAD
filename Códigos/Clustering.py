import pandas as pd 
from pycaret.clustering import *


dados = {
 "Nome": [
 "Ana Souza", "Bruno Lima", "Carla Ferreira", "Diego Costa", "Elisa Martins",
 "Fernando Oliveira", "Gabriela Rocha", "Henrique Almeida", "Isabela Pereira",
"João Silva",
 "Karina Alves", "Leandro Santos", "Mariana Cardoso", "Natan Barbosa", "Olivia Gomes",
 "Paulo Ribeiro", "Quezia Dias", "Rafael Pinto", "Sara Farias", "Tiago Moreira",
 "Ulisses Nunes", "Valentina Ribeiro", "Wagner Sousa", "Ximena Castro", "Yuri Mendes"
 ],
 "Salário": [3200, 4500, 2800, 5200, 3900, 4800, 3100, 5000, 3500, 4200,
 3100, 4700, 3300, 5100, 4000, 4600, 3700, 5500, 3600, 4900,
 3300, 4300, 3400, 5200, 4100],
 "Renda": [3500, 4800, 3000, 5400, 4100, 5000, 3300, 5200, 3700, 4400,
 3300, 4800, 3500, 5300, 4200, 4700, 3900, 5600, 3800, 5000,
 3600, 4500, 3700, 5400, 4300],
 "Valor em Compras": [1500, 2000, 1300, 2100, 1800, 2200, 1600, 2300, 1700,
1900,
 1500, 2050, 1650, 2150, 1850, 2250, 1750, 2350, 1550, 1950,
 1650, 2050, 1750, 2200, 1900],
 "Valor em Ressarcimentos": [200, 250, 150, 300, 220, 270, 180, 310, 210, 260,
 200, 255, 190, 305, 225, 265, 185, 315, 195, 275,
 185, 245, 175, 290, 235],
 "Idade": [28, 35, 24, 42, 30, 38, 27, 40, 32, 36,
 29, 37, 31, 41, 33, 39, 26, 43, 34, 37,
 28, 35, 30, 42, 32],
 "Número de Filhos": [1, 2, 0, 3, 1, 2, 0, 3, 1, 2,
 1, 2, 1, 3, 0, 2, 1, 3, 0, 2,
 1, 2, 1, 3, 0]
} 
df = pd.DataFrame(dados) 

print(df.head()) 

clustering_setup = setup (data = df, normalize = True, silent = True, ignore_features = ['Nome'])
print("Ambiente de PyCaret configurado para clustering!") 

modelos_disponiveis = models()
print(modelos_disponiveis) 

modelo_kmeans = create_model('kmeans', num_clusters=3) 
print(modelo_kmeans) 

plot_model(modelo_kmeans, plot = 'silhouette') 
plot_model(modelo_kmeans, plot = 'elbow') 
plot_model(modelo_kmeans, plot = 'cluster') 
plot_model(modelo_kmeans, plot = 'tsne') 
plot_model(modelo_kmeans, plot = 'distance') 

df_cluster = assign_model(modelo_kmeans) 
print(df_cluster.head()) 

novo_cliente = pd.DataFrame({
 "Nome": ["Carlos Drumond"],
 "Salário": [4000],
 "Renda": [4200],
 "Valor em Compras": [1900],
 "Valor em Ressarcimentos": [250],
 "Idade": [33],
 "Número de Filhos": [1]
}) 

novo_cliente_sem_nome = novo_cliente.drop("Nome", axis=1) 

previsao = predict_model(modelo_kmeans, data = novo_cliente_sem_nome) 

print(previsao) 

