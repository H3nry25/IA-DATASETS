# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             recall_score, precision_score, f1_score, roc_auc_score, 
                             roc_curve, r2_score)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap

print("--- 1. CARGA DE DATOS PREPROCESADOS ---")

# Rutas automáticas para encontrar el dataset limpio
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_dataset = os.path.join(directorio_actual, '..', 'dataset')
# Usamos el archivo 2 que ya tiene la normalización aplicada
ruta_archivo = os.path.join(ruta_dataset, 'train_procesado_2.csv')

try:
    df_ml = pd.read_csv(ruta_archivo)
    print(f"Datos cargados exitosamente de: {ruta_archivo}")
except FileNotFoundError:
    print(f"Error: No se encontró '{ruta_archivo}'. Asegúrate de haber ejecutado el preprocesamiento primero.")
    exit()

# Separamos las características (X) y el objetivo (y)
# En el Titanic, nuestro objetivo es predecir si sobrevivió ('Survived')
X = df_ml.drop(columns=['Survived']).values
y = df_ml['Survived'].values
feature_names = df_ml.drop(columns=['Survived']).columns.tolist()

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Objetivo): {y.shape}\n")

print("--- 2. DIVISIÓN DE DATOS ---")
# Dividimos: 80% para que la IA estudie (entrenamiento) y 20% para el examen final (prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Los datos ya fueron escalados en el preprocesamiento, así que pasamos directo
X_train_scaled = X_train
X_test_scaled = X_test 

print("--- 3. ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA ---")
# class_weight='balanced' ayuda a compensar si hay más muertos que vivos en los datos
model = LogisticRegression(class_weight='balanced', solver='saga', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Imprimimos la ecuación matemática que encontró la regresión
print("\nCoeficientes del modelo (Peso de cada variable):")
print(model.coef_)
print("\nIntercepto del modelo:")
print(model.intercept_)

# Hacemos que el modelo rinda el examen con el 20% de prueba
y_pred = model.predict(X_test_scaled)
# predict_proba nos da el porcentaje exacto de probabilidad para la curva ROC
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] 

print("\n--- 4. EVALUACIÓN DEL MODELO ---")
# Imprime el reporte de notas del modelo (Precisión, Recall, etc.)
print(classification_report(y_test, y_pred))

# Creamos y graficamos la Matriz de Confusión para ver en qué se equivocó
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión: \n", cm)

plt.figure(figsize=(8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Matriz de Confusión - Regresión Logística (Titanic)')
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Realidad', fontsize=12)
plt.show()

# Cálculos de métricas sueltas
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Specificity: ", recall_score(y_test, y_pred, pos_label=0, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred, zero_division=0))

# Curva ROC: Evalúa qué tan bien separa el modelo a los sobrevivientes de los no sobrevivientes
try:
    auc = roc_auc_score(y_test, y_pred_proba)
    print("AUC: ", auc)
    
    plt.figure()
    lw = 2
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Curva ROC (área = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Receiver Operating Characteristic (ROC) - Titanic')
    plt.legend(loc="lower right")
    plt.show()
except ValueError:
    print("AUC: No se puede calcular")

print("R2: ", r2_score(y_test, y_pred))

# --- VISUALIZACIÓN DE COEFICIENTES ---
# Gráfico de barras para ver qué variable impacta más positiva o negativamente en sobrevivir
coefficients = model.coef_
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_names, coefficients[0], color='teal')
ax.set_title('Importancia de las Características (Regresión Logística - Titanic)')
ax.set_xlabel('Coeficientes')
max_val = max(abs(coefficients[0]))
ax.set_xlim(-max_val - 0.5, max_val + 0.5)
plt.tight_layout()
plt.show()

# Guardar el conocimiento del modelo en un archivo en la misma carpeta
ruta_modelo = os.path.join(directorio_actual, 'logistic_regression_titanic.pkl')
joblib.dump(model, ruta_modelo)
print(f"\nModelo guardado exitosamente como '{ruta_modelo}'")

print("\n--- 5. EXPLICABILIDAD CON SHAP ---")
# SHAP analiza las predicciones y explica visualmente por qué el modelo tomó ciertas decisiones
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)