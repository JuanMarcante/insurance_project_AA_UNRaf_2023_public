# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Regresión Lineal Múltiple de Importes 

# Importación de Datos
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

X = dataset.iloc[:, 0:6]
Y = dataset.iloc[:, 6].values.reshape(-1, 1)

# Agregar preprosesamiento de datos

# Dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Ajustar el modelo de Regresión Lineal Múltiple al conjunto de entrenamiento
from sklearn import linear_model
regression = linear_model.LinearRegression()
regression.fit(X_train, Y_train)

# Predicción de los resultados en el conjunto de testing
Y_pred = regression.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE=round(mean_absolute_error(Y_test, Y_pred),2)
MSE=round(mean_squared_error(Y_test, Y_pred),2)
RMSE=round(mean_squared_error(Y_test, Y_pred, squared=False),2)
R2=round(r2_score(Y_test, Y_pred),2)

metrica_MAE = f'El Error Absoluto Medio (MAE) es: {MAE}.'
metrica_MSE = f'El Error Cuadrático Medio (MSE) es: {MSE}.'
metrica_RMSE = f'La Raíz Cuadrada del Error Cuadrático Medio (RMSE) es: {RMSE}.'
metrica_R2 = f'El R^2 es: {R2}.'

print(metrica_MAE)
print(metrica_MSE)
print(metrica_RMSE)
print(metrica_R2)
