from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input
import numpy as np
import pandas as pd
import os
import pylab as pl
from matplotlib import ticker
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Arquivo de entrada


def BuildFeatures(model, file, size):
    arq = open(file, 'r')
    conteudo_entrada = arq.readlines()
    arq.close()

    # Diretorio da base de dados (imagens)
    dir_dataset = './data/'

    img_rows = size
    img_cols = size

    # InceptionV3
    # - weights='imagenet' (inicializa pesos pre-treinado na ImageNet)
    # - include_top=False (nao inclui as fully-connected layers)
    # - input_shape=(299, 299, 3) (DEFAULT) (minimo=75x75)
    #model = InceptionV3(weights='imagenet', include_top=True)

    X = []
    y = []

    print("Loading...", file)
    for i in conteudo_entrada:
        nome, classe = i.split()

        # Load da imagem
        img_path = dir_dataset + nome
        img = image.load_img(img_path, target_size=(img_rows, img_cols))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        # Passa a imagem pela rede
        inception_features = model.predict(img_data)

        # Flatten
        features_np = np.array(inception_features)
        X.append(features_np.flatten())
        y.append(int(classe))

    return X, y


def confusion_matrix_save_png(method, cm):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title("Matriz de Confusao - " + method)
    fig.colorbar(cax)
    ax.set_xticklabels(['', '1', '2', '3', '4', '5', '6',
                        '7', '8', '9', '10', '11', '12'])
    ax.set_yticklabels(['', 'janeiro', 'fevereiro', 'marco', 'abril', 'maio', 'junho',
                        'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    pl.xlabel('Predito')
    pl.ylabel('Valor Real')
    pl.savefig(method + '_cm.png', bbox_inches='tight')


size = 256
model = InceptionV3(include_top=False, weights='imagenet',
                    pooling='avg', input_tensor=Input(shape=(size, size, 3)))
X_train, y_train = BuildFeatures(model, './train.txt', size)
X_test, y_test = BuildFeatures(model, './test.txt', size)

# Treina o classificador
# clfa = GaussianNB()
# clfa = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# clfa = RandomForestClassifier(n_estimators=100)
clfa = LinearDiscriminantAnalysis(solver='lsqr')
clfa = clfa.fit(X_train, y_train)

# testa usando a base de testes
y_pred = clfa.predict(X_test)

# calcula a acurÃ¡cia na base de teste
score = clfa.score(X_test, y_test)

# calcula a matriz de confusÃ£o
matrix = confusion_matrix(y_test, y_pred)

print('Taxa de Reconhecimento:', round(score * 100, 2))
print(matrix)
confusion_matrix_save_png('LDA', matrix)
