from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Importando dados da UCI
iris = fetch_ucirepo(id=53) # Id 53 código da base de dados Iris no UCI databases
data = iris.data

x = data.features

# Normalizando os dados de entrada com Min-Max
scaller = MinMaxScaler()
x_normalized = scaller.fit_transform(x)

dimension = int(input("Escolha a dimensão para exibição dos dados: 2 ou 3: "))

# Reduzindo dimensionalidade dos dados de 4 para a dimensão escolhida, melhorando a visualização 
x_truncated = TruncatedSVD(dimension).fit_transform(x_normalized)


# Criando o modelo a ser treinado
# Numero de clusters settado para 2 pois apresentou o melhor coeficiente de silhueta
# N_init -> Iniciação aleatória dos cluters gerada no nomento do treinamento
k_means = KMeans(n_clusters=2, n_init='auto', random_state=10) 

# Treinando com os dados já normalizados e redimensionados
k_means.fit(x_normalized)
x = x_normalized
# Extraindo os centroids e labels do modelo já treinado
centroids = k_means.cluster_centers_
labels = k_means.labels_

silhouette_svg = silhouette_score(x, k_means.predict(x))
# Gerando grafico para visualização dos dados de acordo com a dimensão escolhida
if dimension == 2:
        plt.scatter(x[:,0], x[:,1], s=50, c=labels)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, alpha=0.75, label='Centróides')
        plt.title(f"Coeficiente de silhueta {silhouette_svg:.2f}")
        plt.show()
elif dimension == 3:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
        x1 = x[:, 0]
        y1 = x[:, 1]
        z1 = x[:, 2]
        scatter = ax.scatter(x1, y1, z1, c=labels, cmap='viridis')
        scatter = ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], marker='x', s=200)
        ax.set_xlabel('Dimensão X')
        ax.set_ylabel('Dimensão Y')
        ax.set_zlabel('Dimensão Z')
        plt.title(f"Coeficiente de silhueta {silhouette_svg:.2f}")
        plt.colorbar(scatter, label='Classes', ticks=range(3), format="Classes")
        plt.show()
else:
        print(silhouette_svg)
