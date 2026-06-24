# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             recall_score, precision_score, f1_score, roc_auc_score, 
                             roc_curve, r2_score)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap

print("--- 1. FASE DE EXTRACCIÓN Y TRANSFORMACIÓN (ETL) ---")
# 1.1 Definimos la ruta donde están los 70 archivos descargados
path = './Diabetes-Data' 
# Buscamos todos los archivos que empiecen con "data-"
all_files = glob.glob(os.path.join(path, "data-*"))

list_df = []
# 1.2 Leemos cada archivo uno por uno y los apilamos
for filename in all_files:
    # sep='\t' porque los datos están separados por tabulaciones. on_bad_lines='skip' ignora filas corruptas.
    df_temp = pd.read_csv(filename, sep='\t', header=None, 
                          names=['Fecha', 'Hora', 'Codigo', 'Valor'], 
                          on_bad_lines='skip')
    # Extraemos el nombre del archivo (ej. data-01) para saber a qué paciente pertenece cada fila
    df_temp['Paciente'] = os.path.basename(filename)
    list_df.append(df_temp)

# 1.3 Unimos todos los pequeños DataFrames en una sola gran tabla
df_completo = pd.concat(list_df, ignore_index=True)
# Aseguramos que la columna 'Valor' sea numérica y borramos los errores (NaN)
df_completo['Valor'] = pd.to_numeric(df_completo['Valor'], errors='coerce')
df_completo = df_completo.dropna(subset=['Valor'])

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (ARMANDO LAS VARIABLES) ---
# 2.1 Definimos qué queremos predecir (Nuestra 'y')
# Filtramos los eventos donde hubo hipoglucemia (código 65) y contamos por paciente y fecha
df_target = df_completo[df_completo['Codigo'] == 65].groupby(['Paciente', 'Fecha']).size().reset_index(name='Hubo_Hipoglucemia')
df_target['Hubo_Hipoglucemia'] = 1 # Si aparece en esta lista, le ponemos 1 (Sí hubo crisis)

# 2.2 Definimos con qué vamos a predecir (Nuestras 'X')
# Solo nos quedamos con las dosis de insulina y mediciones de glucosa clave
df_features = df_completo[df_completo['Codigo'].isin([33, 34, 58, 60, 62])].copy()
# Traducimos los números a nombres legibles
codigo_map = {33: 'Insulina_Regular', 34: 'Insulina_NPH', 58: 'Glucosa_Desayuno', 
              60: 'Glucosa_Almuerzo', 62: 'Glucosa_Cena'}
df_features['Variable'] = df_features['Codigo'].map(codigo_map)

# 2.3 Pivoteamos la tabla: Pasamos las variables de filas a columnas (formato tabular clásico)
df_pivot = df_features.pivot_table(index=['Paciente', 'Fecha'], columns='Variable', values='Valor', aggfunc='mean').reset_index()
# Llenamos los datos faltantes de un día con la mediana general para que el modelo no se caiga
df_pivot = df_pivot.fillna(df_pivot.median(numeric_only=True))

# 2.4 Unimos las características (X) con nuestro objetivo (y)
df_final = pd.merge(df_pivot, df_target, on=['Paciente', 'Fecha'], how='left')
# Si un día no tuvo código 65, se rellenará con NaN. Lo cambiamos a 0 (No hubo crisis)
df_final['Hubo_Hipoglucemia'] = df_final['Hubo_Hipoglucemia'].fillna(0).astype(int)

# --- 3. PREPARACIÓN PARA MACHINE LEARNING ---
# Separamos la tabla limpia en X (datos de entrada) y y (lo que queremos adivinar)
feature_names = ['Glucosa_Almuerzo', 'Glucosa_Cena', 'Glucosa_Desayuno', 'Insulina_NPH', 'Insulina_Regular']
X = df_final[feature_names]
y = df_final['Hubo_Hipoglucemia'].values

print(f"Dimensiones de X: {X.shape}")
print(f"Dimensiones de y: {y.shape}\n")

# Dividimos: 80% para que la IA estudie (entrenamiento) y 20% para el examen final (prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalamos los datos: Hacemos que Glucosa (valores de 100-300) e Insulina (valores de 5-30) estén en la misma escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

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