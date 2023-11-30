# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, 0].values.reshape(-1, 1)
y = dataset.iloc[:, 6].values.reshape(-1, 1)

# Regresión Lineal Simple Edad vs Importe

# Dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# Crear el modelo de regresión lineal simple con el conjunto de entrenamiento
from sklearn import linear_model
regression = linear_model.LinearRegression()
regression.fit(X_train, Y_train)

# Predecir el conjunto de test
Y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Importe del Seguro (Conjunto de Entrenamiento)")
plt.xlabel("Edad")
plt.ylabel("Importe (en $)")
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Importe del Seguro (Conjunto de Prueba)")
plt.xlabel("Edad")
plt.ylabel("Importe (en $)")
plt.show()

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
