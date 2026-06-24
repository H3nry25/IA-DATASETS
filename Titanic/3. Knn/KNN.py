# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- 2. ENTRENAMIENTO DEL MODELO K-NN ---")
# Crea y entrena el modelo (K=3 vecinos)
model = KNeighborsClassifier(n_neighbors=3, p=2, weights='uniform')
model.fit(X_train, y_train)

# Predicciones finales (0 o 1)
y_pred = model.predict(X_test)
# Probabilidades exactas para graficar bien la curva ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("--- 3. EVALUACIÓN DEL MODELO ---")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión: \n", cm)

plt.figure(figsize=(8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - KNN (Titanic)')
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Realidad', fontsize=12)
plt.show()

# Métricas solicitadas
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Specificity: ", recall_score(y_test, y_pred, pos_label=0, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred, zero_division=0))

try:
    auc = roc_auc_score(y_test, y_pred_proba)
    print("AUC: ", auc)
    
    # Curva ROC
    plt.figure()
    lw = 2
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Curva ROC (área = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Receiver Operating Characteristic (ROC) - KNN')
    plt.legend(loc="lower right")
    plt.show()
except ValueError:
    print("AUC: No se puede calcular (posiblemente falta una clase en los datos de prueba)")

print("R2: ", r2_score(y_test, y_pred))

# Guardar y cargar modelo en la misma carpeta
ruta_modelo = os.path.join(directorio_actual, 'knn_model_titanic.pkl')
joblib.dump(model, ruta_modelo)
print(f"\nModelo guardado exitosamente como '{ruta_modelo}'")

print("\n--- 4. EXPLICABILIDAD CON SHAP ---")
# SHAP es pesado con KNN, usamos una muestra como en tu código original
sample_size = min(50, len(X_train), len(X_test)) 
X_train_sample = shap.sample(X_train, sample_size)
X_test_sample = shap.sample(X_test, sample_size)

# Función que solo devuelva la probabilidad de la clase 1
predict_fn = lambda x: model.predict_proba(x)[:, 1]

# Le pasamos nuestra nueva función al KernelExplainer
explainer = shap.KernelExplainer(predict_fn, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names)