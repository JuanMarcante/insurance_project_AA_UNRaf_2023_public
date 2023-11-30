# Naive Bayes

# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importación de datos
dataset = pd.read_csv('insurance.csv')

# Tratamiento de variables categóricas
dataset.replace({'sex':{'female':0,'male':1}}, inplace=True)

X = dataset.iloc[:, [1, 6]].values
Y = dataset.iloc[:, 4].values

# Tratamiento de variables categóricas
dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# Dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicción de los resultados con el conjunto de testing
Y_pred = classifier.predict(X_test)

# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label='yes')
recall = recall_score(Y_test, Y_pred, pos_label='yes')
f1_score_m = f1_score(Y_test, Y_pred, pos_label='yes')

metric_accuracy = f'La métrica Accuracy es de {round(accuracy, 2)}.'
metric_precision = f'La métrica Precision es de {round(precision, 2)}.'
metric_recall = f'La métrica Recall es de {round(recall, 2)}.'
metric_f1_score = f'La métrica F1 Score es de {round(f1_score_m, 2)}.'

print(metric_accuracy)
print(metric_precision)
print(metric_recall)
print(metric_f1_score)