from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import sklearn.metrics as metrics
import numpy as np

# Importando base de dados da UCI
wine = fetch_ucirepo(id=109)  # 109 ID da base de dados Wine do UCI
data = wine.data
datas = data.original

x = np.array(datas.drop(['class'], axis=1))
y = np.array(datas['class'])

# y = LabelEncoder().fit_transform(y) # Útil para padronizar classes que não estejam padronizadas com 0, 1, 2... por exemplo

# Utilizando a normalização MinMax para os dados de entrada ou features
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
 
# Utilizando o metódo Hold-out para separar o conjunto de dados que vai para teste e o que vai para treinamento
# test_size -> Tamanho do conjunto de dados de teste em decimal. Ex: 50% -> 0.5
# random_state -> Serve para escolher de forma 'aleatória' garantindo que em múltiplas chamadas não sejam escolhidos os mesmos dados

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.33, random_state=50)

# Criando modelo MLP que será treinado com os dados de treinamento
# activation -> Função de ativação que será utilizada no treinamento do modelo
# max_iter -> Número máximo de interações que o modelo irá fazer ao treinar
# hidden_layer_sizes -> Número de camadas ocultas do modelo
# alpha -> Taxa de regularização que será aplicada aos pesos no treinamento
mlp = MLPClassifier(activation='logistic', max_iter=1000,
                    hidden_layer_sizes=(5,), alpha=0.001, solver='lbfgs')

# Treinando o modelo com os dados de treinamento
mlp.fit(x_train, y_train)

# Fazendo as previsões com o conjunto de dados de teste
predictions = mlp.predict(x_test)

# Exibindo o resultado da Rede Neural Artificial e as saídas reais
print("\nSaídas da RNA:\n", predictions)
print("\nSaídas esperadas:\n", y_test, "\n")

# Metódo de avaliação de acurácia entre as saídas previstas pela RNA e as saídas reais
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy:.2f}\n")

# Aplicando o metódo de avaliação Cross-Validation com 10 folds
# cv -> Número de folds
cross_val_scores = cross_val_score(
    mlp, x_scaled, y, cv=10)

# Exiba as pontuações de cada fold e a média das pontuações
print("Pontuações de validação cruzada:", cross_val_scores)
media_cross_val_scores = cross_val_scores.mean()
print(f"Média das pontuações de validação cruzada:{media_cross_val_scores:.2f}\n")

# Criando e exibindo a matriz de confusão para avaliação dos resultados
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(confusion_matrix)
