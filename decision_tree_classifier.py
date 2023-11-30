import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Cargar datos desde el archivo CSV
insurance_data = pd.read_csv('insurance.csv')


# Separar los datos según las etiquetas de la variable objetivo
no_fumadores = insurance_data[insurance_data["smoker"] == 'no']
fumadores = insurance_data[insurance_data["smoker"] == 'yes']

# Gráfico de dispersión Edad vs Costo Médico
plt.figure(figsize=(10, 6))
plt.scatter(no_fumadores["age"], no_fumadores["charges"],
            label="No Fumadores", marker="o", color="skyblue", s=100, alpha=0.7)
plt.scatter(fumadores["age"], fumadores["charges"],
            label="Fumadores", marker="o", color="lightcoral", s=100, alpha=0.7)

# Personalizar el gráfico
plt.title('Relación entre Edad y Costos Médicos', fontsize=20)
plt.xlabel('Edad', fontsize=16)
plt.ylabel('Costos Médicos', fontsize=16)
plt.legend()
plt.grid(True)

# Gráfico de dispersión BMI vs Costo Médico
plt.figure(figsize=(10, 6))
plt.scatter(no_fumadores["bmi"], no_fumadores["charges"],
            label="No Fumadores", marker="o", color="skyblue", s=100, alpha=0.7)
plt.scatter(fumadores["bmi"], fumadores["charges"],
            label="Fumadores", marker="o", color="lightcoral", s=100, alpha=0.7)

# Personalizar el gráfico
plt.title('Relación entre BMI y Costos Médicos', fontsize=20)
plt.xlabel('BMI', fontsize=16)
plt.ylabel('Costos Médicos', fontsize=16)
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()

# Variables de interés
edades = insurance_data['age']
ibms = insurance_data['bmi']

# Normalizar las variables para obtener distribuciones de probabilidad
prob_edades = edades.value_counts(normalize=True, sort=False)
prob_ibms = ibms.value_counts(normalize=True, sort=False)

# Calcular la entropía para cada variable
entropia_edades = entropy(prob_edades, base=2)
entropia_ibms = entropy(prob_ibms, base=2)

# Mostrar resultados
print("Distribución de probabilidad de edades:")
print(prob_edades)
print("Entropía para edades:", entropia_edades)

print("\nDistribución de probabilidad de índice de masa corporal (BMI):")
print(prob_ibms)
print("Entropía para BMI:", entropia_ibms)

# Variables de interés
caracteristicas = insurance_data[['age', 'bmi']]
etiquetas = insurance_data['smoker']  # Cambia 'smoker' a la columna que representa la variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
datos_entrenamiento, datos_prueba, clase_entrenamiento, clase_prueba = train_test_split(
    caracteristicas,
    etiquetas,
    test_size=0.23,
    random_state=42  # Para reproducibilidad, puedes cambiar este valor o quitarlo
    )

# Mostrar la forma de los conjuntos de entrenamiento y prueba
print("Forma del conjunto de entrenamiento:", datos_entrenamiento.shape, clase_entrenamiento.shape)
print("Forma del conjunto de prueba:", datos_prueba.shape, clase_prueba.shape)

# Creamos el árbol indicando que lo haremos con el criterio "entropía" y con una profundidad de 2
arbol_decision = DecisionTreeClassifier(criterion="entropy",
                                        max_depth=5,
                                        splitter = 'best',
                                        min_samples_split = 2)

# Guardamos en la variable 'arbol' el modelo creado
arbol = arbol_decision.fit(datos_entrenamiento, clase_entrenamiento)

# Mostramos en formato texto el árbol creado
print(tree.export_text(arbol, feature_names=["Edad", "BMI"]))

# Mostramos en formato gráfico el árbol creado
plt.figure(figsize=(12, 6))
tree.plot_tree(arbol, feature_names=["Edad", "BMI"],
               class_names=["No Fumador", "Fumador"],
               filled=True)
plt.show()

parametros_arbol_decision = arbol_decision.get_params()

# print("Nuevo paciente", arbol_decision.predict([[70, 150]]))
# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = datos_entrenamiento
X_test = datos_prueba
Y_train = clase_entrenamiento
Y_test = clase_prueba
Y_pred = arbol_decision.predict(datos_prueba)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label='no')
recall = recall_score(Y_test, Y_pred, pos_label='no')
f1_score_m = f1_score(Y_test, Y_pred, pos_label='no')

metric_accuracy = f'La métrica Accuracy es de {round(accuracy, 4)}.'
metric_precision = f'La métrica Precision es de {round(precision, 4)}.'
metric_recall = f'La métrica Recall es de {round(recall, 4)}.'
metric_f1_score = f'La métrica F1 Score es de {round(f1_score_m, 4)}.'

print(metric_accuracy)
print(metric_precision)
print(metric_recall)
print(metric_f1_score)