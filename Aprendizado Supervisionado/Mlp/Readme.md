# Inteligência Artificial
### Utilizando MLP para resolver problemas de classificação

Este programa trata-se de um trabalho avaliativo feito pelo aluno [Lucas José](https://github.com/yamatosz), para a disciplina de Inteligência Artificial do curso de Ciência da Computação da Universidade Federal do Tocantins, ministrada pelo Professor Doutor Alexandre Tadeu Rossini.

A base de dados utilizada para este projeto foi a [Wine](https://archive.ics.uci.edu/dataset/109/wine), da [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)

Aqui utilizamos de ferramentas das bibliotecas sci-kit learn, e numpy para resolver um problema de classificação por meio de uma RNA MLP. 

Para o metódo de treinamento e avaliação foi implementado os metódos ***Hold-out*** e ***Cross-validation***. Como métrica de avaliação da qualidade do aprendizado a ***acurácia*** e ***matriz de confusão***

Para executar o programa, tenha certeza de ter instalado em sua máquina as bibliotecas necessárias.

Utilize o seguinte comando para instalar as bibliotecas necessárias para o programa funcionar:
```
pip install requirements.txt
```
E em seguida, execute o comando a seguir para executar o programa:
```
py mlp.py
```