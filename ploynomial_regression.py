# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importación de datos
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, 0].values.reshape(-1, 1)
Y = dataset.iloc[:, 6].values.reshape(-1, 1)

# Dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Ajustar la Regresión Lineal al dataset
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly_train = poly_reg.fit_transform(X_train)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly_train, Y_train)

# Predicción de nuestros modelos en el conjunto de prueba
X_poly_test = poly_reg.transform(X_test)
Y_pred = lin_reg_2.predict(X_poly_test)

# Visualización del modelo
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title("Modelo Polinómico")
plt.xlabel("Edad")
plt.ylabel("Importe (en $)")
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
MAE=round(mean_absolute_error(Y_test, Y_pred),2)
MSE=round(mean_squared_error(Y_test, Y_pred),2)
RMSE=round(mean_squared_error(Y_test, Y_pred, squared=False),2)

metrica_MAE = f'El Error Absoluto Medio (MAE) es: {MAE}.'
metrica_MSE = f'El Error Cuadrático Medio (MSE) es: {MSE}.'
metrica_RMSE = f'La Raíz Cuadrada del Error Cuadrático Medio (RMSE) es: {RMSE}.'

print(metrica_MAE)
print(metrica_MSE)
print(metrica_RMSE)