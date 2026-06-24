# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             recall_score, precision_score, f1_score, roc_auc_score, 
                             roc_curve, r2_score)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap

print("--- 1. FASE DE EXTRACCIÓN Y TRANSFORMACIÓN (ETL) ---")
# Ruta a tu carpeta de datos según tu terminal
path = './Diabetes-Data' 
all_files = glob.glob(os.path.join(path, "data-*"))

list_df = []
# Leer y concatenar los 70 archivos
for filename in all_files:
    # Ignoramos lineas corruptas si las hay con on_bad_lines='skip'
    df_temp = pd.read_csv(filename, sep='\t', header=None, 
                          names=['Fecha', 'Hora', 'Codigo', 'Valor'], 
                          on_bad_lines='skip')
    df_temp['Paciente'] = os.path.basename(filename) # Guardar de qué paciente es
    list_df.append(df_temp)

df_completo = pd.concat(list_df, ignore_index=True)
df_completo['Valor'] = pd.to_numeric(df_completo['Valor'], errors='coerce')
df_completo = df_completo.dropna(subset=['Valor'])

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING) ---
# Objetivo (y): 1 si el paciente tuvo hipoglucemia (Código 65) ese día, 0 si no.
df_target = df_completo[df_completo['Codigo'] == 65].groupby(['Paciente', 'Fecha']).size().reset_index(name='Hubo_Hipoglucemia')
df_target['Hubo_Hipoglucemia'] = 1 

# Características (X): Dosis de Insulina (33, 34) y Glucosa (58, 60, 62)
df_features = df_completo[df_completo['Codigo'].isin([33, 34, 58, 60, 62])].copy()
codigo_map = {33: 'Insulina_Regular', 34: 'Insulina_NPH', 58: 'Glucosa_Desayuno', 
              60: 'Glucosa_Almuerzo', 62: 'Glucosa_Cena'}
df_features['Variable'] = df_features['Codigo'].map(codigo_map)

# Agrupar por día y paciente
df_pivot = df_features.pivot_table(index=['Paciente', 'Fecha'], columns='Variable', values='Valor', aggfunc='mean').reset_index()
# Llenar los días que no hubo medición de alguna variable con la mediana para no perder datos
df_pivot = df_pivot.fillna(df_pivot.median(numeric_only=True))

# Unir características con la variable objetivo
df_final = pd.merge(df_pivot, df_target, on=['Paciente', 'Fecha'], how='left')
df_final['Hubo_Hipoglucemia'] = df_final['Hubo_Hipoglucemia'].fillna(0).astype(int)

# --- 3. PREPARACIÓN PARA MACHINE LEARNING ---
feature_names = ['Glucosa_Almuerzo', 'Glucosa_Cena', 'Glucosa_Desayuno', 'Insulina_NPH', 'Insulina_Regular']
X = df_final[feature_names].values
y = df_final['Hubo_Hipoglucemia'].values

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Objetivo): {y.shape}\n")

# Divide el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normaliza los datos
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("--- 4. ENTRENAMIENTO DEL MODELO K-NN ---")
# Crea y entrena el modelo
model = KNeighborsClassifier(n_neighbors=3, p=2, weights='uniform')
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

print("--- 5. EVALUACIÓN DEL MODELO ---")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)

plt.figure(figsize = (8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Predicción de Hipoglucemia')
plt.xlabel('Prediction', fontsize = 12)
plt.ylabel('Real', fontsize = 12)
plt.show()

# Métricas solicitadas
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Specificity: ", recall_score(y_test, y_pred, pos_label=0, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred, zero_division=0))

try:
    auc = roc_auc_score(y_test, y_pred)
    print("AUC: ", auc)
    
    # Curva ROC
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
    print("AUC: No se puede calcular (posiblemente falta una clase en los datos de prueba)")

print("R2: ", r2_score(y_test, y_pred))

# Guardar y cargar modelo
joblib.dump(model, 'knn_model_diabetes.pkl')
loaded_model = joblib.load('knn_model_diabetes.pkl')

print("\n--- 6. EXPLICABILIDAD CON SHAP ---")
# SHAP es pesado, usamos una muestra como en tu código original
sample_size = min(50, len(X_train), len(X_test)) 
X_train_sample = shap.sample(X_train, sample_size)
X_test_sample = shap.sample(X_test, sample_size)

# SOLUCIÓN: Creamos una función que solo devuelva la probabilidad de la clase 1
predict_fn = lambda x: model.predict_proba(x)[:, 1]

# Le pasamos nuestra nueva función al KernelExplainer
explainer = shap.KernelExplainer(predict_fn, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

# Como ahora solo calculamos una salida, pasamos shap_values directo (sin el [1])
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names)