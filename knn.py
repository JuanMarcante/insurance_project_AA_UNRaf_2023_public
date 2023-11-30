# K-Nearest Neighbors (K-NN)

# Predecir si una persona es fumadora o no en base a su edad, bmi, sexo y región

#Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importación de datos
dataset = pd.read_csv('insurance.csv')

# Tratamiento de variables categóricas
dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
dataset.replace({'region':{'southeast':0,
                           'southwest':1,
                           'northeast':2,
                           'northwest':3}},
                            inplace=True)
age_c = []
bmi_c = []

for i in dataset['bmi']:
    if i < 18.5:
        bmi_c.append(0)
    elif i < 25:
        bmi_c.append(1)
    elif i < 30:
        bmi_c.append(2)
    else:
        bmi_c.append(3)
        
for i in dataset['age']:
    if i < 13:
        age_c.append(0)
    elif i < 19:
        age_c.append(1)
    elif i < 30:
        age_c.append(2)
    elif i < 60:
        age_c.append(3)
    else:
        age_c.append(4)

dataset['age_c'] = age_c
dataset['bmi_c'] = bmi_c

X = dataset.iloc[:, [1,5,7,8]].values
Y = dataset.iloc[:, 4].values

# Dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el modelo de regresión logística al conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='brute', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
classifier.fit(X_train, Y_train)

# Predicción de los resultados con el conjunto de testing
Y_pred = classifier.predict(X_test)

# Elaborar la matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
ac = accuracy_score(Y_test, Y_pred)

# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label=1)
recall = recall_score(Y_test, Y_pred, pos_label=1)
f1_score_m = f1_score(Y_test, Y_pred, pos_label=1)

metric_accuracy = f'La métrica Accuracy es de {round(accuracy, 2)}.'
metric_precision = f'La métrica Precision es de {round(precision, 2)}.'
metric_recall = f'La métrica Recall es de {round(recall, 2)}.'
metric_f1_score = f'La métrica F1 Score es de {round(f1_score_m, 2)}.'

print(metric_accuracy)
print(metric_precision)
print(metric_recall)
print(metric_f1_score)