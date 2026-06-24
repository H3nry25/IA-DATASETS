# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             recall_score, precision_score, f1_score, roc_auc_score, 
                             roc_curve, r2_score)
from sklearn.tree import plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap

print("--- 1. CARGA DE DATOS PREPROCESADOS ---")

# Rutas automáticas para encontrar el dataset limpio
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_dataset = os.path.join(directorio_actual, '..', 'dataset')
ruta_archivo = os.path.join(ruta_dataset, 'train_procesado_2.csv')

try:
    df_ml = pd.read_csv(ruta_archivo)
    print(f"Datos cargados exitosamente de: {ruta_archivo}")
except FileNotFoundError:
    print(f"Error: No se encontró '{ruta_archivo}'. Asegúrate de haber ejecutado el preprocesamiento primero.")
    exit()

# Separamos las características (X) y el objetivo (y) para el Titanic
X = df_ml.drop(columns=['Survived']).values
y = df_ml['Survived'].values
feature_names = df_ml.drop(columns=['Survived']).columns.tolist()

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Objetivo): {y.shape}\n")

print("--- 2. DIVISIÓN DE DATOS ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- 3. ENTRENAMIENTO DEL RANDOM FOREST ---")
# n_estimators=15 significa que creará 15 árboles distintos y tomará la decisión por mayoría de votos
model = RandomForestClassifier(n_estimators=15, max_depth=4, criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- 4. EVALUACIÓN DEL MODELO ---")
print(classification_report(y_test, y_pred))

# Matriz de confusión:
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión: \n", cm)

plt.figure(figsize=(8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title('Matriz de Confusión - Random Forest (Titanic)')
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Realidad', fontsize=12)
plt.show()

# Métricas sueltas:
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Specificity: ", recall_score(y_test, y_pred, pos_label=0, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred, zero_division=0))

# Curva ROC y AUC
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
    plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
    plt.legend(loc="lower right")
    plt.show()
except ValueError:
    print("AUC: No se puede calcular")

print("R2: ", r2_score(y_test, y_pred))

# --- VISUALIZAR UN ÁRBOL DEL BOSQUE ---
print("\nGenerando gráfico de un Árbol del Bosque Aleatorio...")
# Seleccionamos el primer árbol (índice 0) de los 15 que se crearon
estimator = model.estimators_[0]

fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(estimator, 
          feature_names=feature_names, 
          class_names=['Murió', 'Sobrevivió'], 
          filled=True, 
          rounded=True, 
          ax=ax,
          fontsize=10)
plt.title("Ejemplo de 1 de los 15 árboles en el Random Forest")
plt.show()

# --- IMPORTANCIA DE LAS VARIABLES ---
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=True)

print("\nImportancia de las variables combinada (Los 15 árboles opinan):")
print(feature_importances.sort_values(by='Importancia', ascending=False))

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Variable'], feature_importances['Importancia'], color='purple')
plt.xlabel('Importancia (Entropía)')
plt.ylabel('Variables')
plt.title('Importancia de las características (Random Forest)')
plt.show()

# --- GUARDAR MODELO ---
ruta_modelo = os.path.join(directorio_actual, 'random_forest_titanic.pkl')
joblib.dump(model, ruta_modelo)
print(f"\nModelo guardado exitosamente como '{ruta_modelo}'")

print("\n--- 5. EXPLICABILIDAD CON SHAP ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Adaptamos para manejar cómo SHAP devuelve los valores según la versión
if isinstance(shap_values, list):
    valores_graficar = shap_values[1]
else:
    # A veces en Random Forest SHAP devuelve un array 3D
    if len(shap_values.shape) == 3:
        valores_graficar = shap_values[:, :, 1]
    else:
        valores_graficar = shap_values

shap.summary_plot(valores_graficar, X_test, feature_names=feature_names)