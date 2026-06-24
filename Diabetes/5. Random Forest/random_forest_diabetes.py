# -*- coding: utf-8 -*-
"""

@author: IVAN

"""

# Random Forest for Diabetes (Predicción de Hipoglucemia)

import os

# Ruta absoluta al directorio donde vive este script
_DIR = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

print("--- 1. CARGA DE DATOS PREPROCESADOS ---")
# Construimos la ruta al CSV relativa a la ubicación de este script (funciona
# sin importar desde qué directorio lo ejecute VS Code o la terminal)
_CSV_PATH = os.path.join(_DIR, '..', '1. Preprocesamiento', 'diabetes_preprocessed.csv')
_CSV_PATH = os.path.normpath(_CSV_PATH)

try:
    df_ml = pd.read_csv(_CSV_PATH)
    print(f"Datos cargados exitosamente de '{_CSV_PATH}'.")
except FileNotFoundError:
    print(f"Error: No se encontró '{_CSV_PATH}'. Por favor, ejecuta 'preprocess_data.py' primero.")
    exit()

# Separamos las características (X) y el objetivo (y)
# X: variables predictoras (glucosas e insulinas)
# y: variable objetivo (0 = no hipoglucemia, 1 = sí hubo hipoglucemia)
X = df_ml.drop(columns=['Hubo_Hipoglucemia']).values  # 3800x5
y = df_ml['Hubo_Hipoglucemia'].values                 # 3800x1
feature_names = df_ml.drop(columns=['Hubo_Hipoglucemia']).columns.tolist()

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Objetivo): {y.shape}\n")

# Las clases: 0 = No hubo hipoglucemia (negativo); 1 = Sí hubo hipoglucemia (positivo/caso de interés)
# No se invierte y porque 1 ya representa el caso positivo (hipoglucemia) que nos interesa detectar.

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Normaliza los datos para que todas las características tengan una escala similar
# Nota: el CSV ya viene preprocesado con MinMaxScaler, pero aplicamos uno nuevo
# sobre el split de entrenamiento para no filtrar información del test set (data leakage).
scaler = MinMaxScaler(feature_range=(0, 1))  # [0, 1]
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Crea y entrena el modelo Random Forest
# n_estimators: número de árboles en el bosque
# criterion: entropy, gini
# class_weight='balanced': compensa el desbalance de clases (hipoglucemia es un evento raro)
model = RandomForestClassifier(n_estimators=15, max_depth=4, criterion='entropy',
                               class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)

# Convierte las probabilidades en etiquetas binarias (0 o 1)
y_pred = (y_pred > 0.5)

# Muestra el informe de evaluación del modelo entrenado
print(classification_report(y_test, y_pred))

# Matriz de confusión:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)
# gráfica cm
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Hipoglucemia', 'Hipoglucemia'],
            yticklabels=['No Hipoglucemia', 'Hipoglucemia'])
plt.title('Matriz de Confusión - Random Forest Diabetes')
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.show()

# Exactitud:
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("accuracy: ", acc)

# Sensibilidad:
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, zero_division=0)
print("recall: ", recall)

# Precisión:
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, zero_division=0)
print("precision: ", precision)

# Especificidad
# 'specificity' is just a special case of 'recall'.
# specificity is the recall of the negative class
specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
print("specificity: ", specificity)

# Puntuación F1:
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, zero_division=0)
print("f1 score: ", f1)

# Área bajo la curva:
from sklearn.metrics import roc_auc_score
try:
    auc = roc_auc_score(y_test, y_pred)
    print("auc: ", auc)

    # Curva ROC
    from sklearn.metrics import roc_curve
    plt.figure()
    lw = 2
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
except ValueError:
    print("AUC: No se puede calcular (solo una clase en y_test o y_pred)")

# R Score (R^2 coefficient of determination)
from sklearn.metrics import r2_score
R = r2_score(y_test, y_pred)
print("R2: ", R)


# Visualizar un árbol de decisión usando matplotlib
from sklearn.tree import plot_tree
# Seleccionar el primer árbol (0) del modelo Random Forest.
# Visualizar el bosque completo no es práctico por la gran cantidad de árboles.
estimator = model.estimators_[0]
# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(24, 20))
# Nombres de clases para la visualización del árbol
class_names = ['No Hipoglucemia', 'Hipoglucemia']
# Dibujar el árbol de decisión (primer árbol del bosque)
plot_tree(estimator,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          ax=ax)
plt.title('Árbol #0 del Random Forest - Predicción de Hipoglucemia en Diabetes')
# Mostrar la gráfica
plt.show()

# Calcular y visualizar la importancia de las variables en la predicción del modelo
importances = model.feature_importances_

# Crear un DataFrame para visualizar las importancias
feature_importances = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

print(feature_importances)

# Visualizar las importancias de las variables
plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Variable'], feature_importances['Importancia'])
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.title('Importancia de las variables - Random Forest Diabetes')
plt.gca().invert_yaxis()
plt.show()

# Guardar el modelo a un archivo
import joblib
joblib.dump(model, 'random_forest_diabetes.pkl')
# Cargar el modelo desde el archivo
loaded_model = joblib.load('random_forest_diabetes.pkl')
# Hacer predicciones con el modelo cargado
y_pred = model.predict(X_test)

# SHAP (SHapley Additive exPlanations) para explicar las predicciones de un modelo de machine learning.
# import shap # pip install shap
# # Crear un explainer de SHAP usando el conjunto de entrenamiento
# explainer = shap.TreeExplainer(model)
# # Obtener las explicaciones SHAP para el conjunto de prueba
# shap_values = explainer.shap_values(X_test)
# # Proporciona una visión general de la importancia de las características y su impacto en las predicciones.
# # shap_values[1] corresponde a la clase 1 (Hipoglucemia).
# shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
