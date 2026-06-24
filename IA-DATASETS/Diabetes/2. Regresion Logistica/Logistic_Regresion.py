# -*- coding: utf-8 -*-
import os
import glob
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
try:
    # Intenta cargar asumiendo que se ejecuta desde la carpeta de Regresion Logistica
    df_ml = pd.read_csv('../1. Preprocesamiento/diabetes_preprocessed.csv')
    print("Datos cargados exitosamente de '../1. Preprocesamiento/diabetes_preprocessed.csv'.")
except FileNotFoundError:
    try:
        # Intenta cargar asumiendo que se ejecuta desde la raíz del proyecto
        df_ml = pd.read_csv('IA-DATASETS/Diabetes/1. Preprocesamiento/diabetes_preprocessed.csv')
        print("Datos cargados exitosamente de 'IA-DATASETS/Diabetes/1. Preprocesamiento/diabetes_preprocessed.csv'.")
    except FileNotFoundError:
        try:
            # Intenta cargar si el archivo está en la misma carpeta
            df_ml = pd.read_csv('diabetes_preprocessed.csv')
            print("Datos cargados exitosamente de 'diabetes_preprocessed.csv'.")
        except FileNotFoundError:
            print("Error: No se encontró 'diabetes_preprocessed.csv'. Por favor, ejecuta 'preprocess_data.py' primero.")
            exit()

# Separamos las características (X) y el objetivo (y)
X = df_ml.drop(columns=['Hubo_Hipoglucemia']).values
y = df_ml['Hubo_Hipoglucemia'].values
feature_names = df_ml.drop(columns=['Hubo_Hipoglucemia']).columns.tolist()

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Objetivo): {y.shape}\n")

# Dividimos: 80% para que la IA estudie (entrenamiento) y 20% para el examen final (prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Los datos ya fueron escalados en el preprocesamiento usando MinMaxScaler
X_train_scaled = X_train
X_test_scaled = X_test 

print("--- 4. ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA ---")
# AQUÍ ESTÁ LA MAGIA: class_weight='balanced'
# Esto obliga al algoritmo a prestarle más atención a los casos raros (el 1, la hipoglucemia)
# y a no irse por lo fácil respondiendo siempre 0.
model = LogisticRegression(class_weight='balanced', solver='saga', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Imprimimos la ecuación matemática que encontró la regresión
print("\nCoeficientes del modelo (Peso de cada variable):")
print(model.coef_)
print("\nIntercepto del modelo:")
print(model.intercept_)

# Hacemos que el modelo rinda el examen con el 20% de prueba
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\n--- 5. EVALUACIÓN DEL MODELO ---")
# Imprime el reporte de notas del modelo (Precisión, Recall, etc.)
print(classification_report(y_test, y_pred))

# Creamos y graficamos la Matriz de Confusión para ver en qué se equivocó
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)

plt.figure(figsize = (8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title('Matriz de Confusión - Regresión Logística Balanceada')
plt.xlabel('Prediction', fontsize = 12)
plt.ylabel('Real', fontsize = 12)
plt.show()

# Cálculos de métricas sueltas
print("accuracy: ", accuracy_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred, zero_division=0))
print("precision: ", precision_score(y_test, y_pred, zero_division=0))
print("specificity: ", recall_score(y_test, y_pred, pos_label=0, zero_division=0))
print("f1 score: ", f1_score(y_test, y_pred, zero_division=0))

# Curva ROC: Evalúa qué tan bien separa el modelo a los sanos de los enfermos
try:
    auc = roc_auc_score(y_test, y_pred)
    print("auc: ", auc)
    
    plt.figure()
    lw = 2
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
except ValueError:
    print("AUC: No se puede calcular")

print("R2: ", r2_score(y_test, y_pred))

# --- VISUALIZACIÓN DE COEFICIENTES ---
# Gráfico de barras para ver qué variable impacta más positiva o negativamente
coefficients = model.coef_
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_names, coefficients[0], color='teal')
ax.set_title('Importancia de las Características (Regresión Logística)')
ax.set_xlabel('Coeficientes')
max_val = max(abs(coefficients[0]))
ax.set_xlim(-max_val - 0.5, max_val + 0.5)
plt.tight_layout()
plt.show()

# Guardar el conocimiento del modelo en un archivo
joblib.dump(model, 'logistic_regression_diabetes.pkl')
loaded_model = joblib.load('logistic_regression_diabetes.pkl')

print("\n--- 6. EXPLICABILIDAD CON SHAP ---")
# SHAP analiza las predicciones y explica visualmente por qué el modelo tomó ciertas decisiones
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)