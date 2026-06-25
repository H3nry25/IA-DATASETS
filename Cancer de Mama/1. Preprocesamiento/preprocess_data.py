# -*- coding: utf-8 -*-
"""

@author: IVAN

"""

# Preprocesamiento del dataset Breast Cancer
# Genera: breast_cancer_preprocessed.csv

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta absoluta al directorio donde vive este script
_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
print("--- 1. LECTURA DE DATOS ---")
# =============================================================================
# Carga el conjunto de datos Breast Cancer desde sklearn
# Fuente: UCI Machine Learning Repository
# 569 muestras, 30 características numéricas, 1 variable objetivo (maligno/benigno)
dataset = load_breast_cancer()

# Crear DataFrame con nombres de columnas
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Agregar la columna objetivo
# En sklearn: 0=maligno, 1=benigno
# Invertimos para usar el estándar médico: 1=maligno (caso crítico), 0=benigno
df['Diagnostico'] = 1 - dataset.target

print(f"Dataset cargado exitosamente.")
print(f"Forma del dataset: {df.shape}  ({df.shape[0]} muestras, {df.shape[1]-1} características + 1 objetivo)")
print(f"\nColumnas:\n{list(df.columns)}")

# =============================================================================
print("\n--- 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---")
# =============================================================================

print("\nPrimeras 5 filas del dataset:")
print(df.head())

print("\nInformación general del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe().T)

# Distribución de la variable objetivo
print("\nDistribución de la variable objetivo 'Diagnostico':")
conteo = df['Diagnostico'].value_counts()
print(f"  Benigno  (0): {conteo.get(0, 0)} muestras ({conteo.get(0, 0)/len(df)*100:.1f}%)")
print(f"  Maligno  (1): {conteo.get(1, 0)} muestras ({conteo.get(1, 0)/len(df)*100:.1f}%)")

# Verificar valores nulos
print(f"\nValores nulos por columna:\n{df.isnull().sum().sum()} nulos en total (dataset limpio por defecto)")

# --- Gráfica 1: Distribución de la clase objetivo ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Diagnostico', hue='Diagnostico', data=df,
              palette=['steelblue', 'tomato'], legend=False)
plt.title('Distribución del Diagnóstico\n(0 = Benigno, 1 = Maligno)')
plt.xlabel('Diagnóstico')
plt.ylabel('Cantidad de muestras')
plt.xticks([0, 1], ['Benigno (0)', 'Maligno (1)'])
plt.tight_layout()
plt.show()

# --- Gráfica 2: Boxplots de las 10 primeras características por diagnóstico ---
feature_names = dataset.feature_names.tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, feat in enumerate(feature_names[:10]):
    sns.boxplot(x='Diagnostico', y=feat, data=df, ax=axes[i],
                palette=['steelblue', 'tomato'])
    axes[i].set_title(feat, fontsize=9)
    axes[i].set_xlabel('')
plt.suptitle('Distribución de las primeras 10 características por Diagnóstico', fontsize=12)
plt.tight_layout()
plt.show()

# --- Gráfica 3: Mapa de calor de correlación entre características principales ---
plt.figure(figsize=(14, 10))
corr_matrix = df[feature_names[:15]].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor de Correlación (primeras 15 características)')
plt.tight_layout()
plt.show()

# --- Gráfica 4: Histogramas de las 6 características más relevantes ---
top_features = ['mean radius', 'mean texture', 'mean perimeter',
                'mean area', 'mean smoothness', 'mean concavity']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    axes[i].hist(df[df['Diagnostico'] == 0][feat], bins=30, alpha=0.6,
                 color='steelblue', label='Benigno')
    axes[i].hist(df[df['Diagnostico'] == 1][feat], bins=30, alpha=0.6,
                 color='tomato', label='Maligno')
    axes[i].set_title(feat)
    axes[i].legend()
plt.suptitle('Distribución de características clave por Diagnóstico', fontsize=12)
plt.tight_layout()
plt.show()

# =============================================================================
print("\n--- 3. LIMPIEZA DE DATOS ---")
# =============================================================================
# El dataset de Breast Cancer de sklearn ya viene limpio (sin nulos ni duplicados).
# Se verifica de forma explícita para documentar el proceso.

print(f"Valores nulos: {df.isnull().sum().sum()}")
print(f"Filas duplicadas: {df.duplicated().sum()}")
print("Dataset limpio. No se requiere eliminación de registros.")

# =============================================================================
print("\n--- 4. PREPROCESAMIENTO FINAL Y ESTANDARIZACIÓN ---")
# =============================================================================

# Separar características y variable objetivo
X = df[feature_names].copy()
y = df['Diagnostico'].copy()

print(f"Características (X): {X.shape}  —  {X.shape[1]} variables")
print(f"Objetivo      (y): {y.shape}  —  {y.value_counts().to_dict()}")

# Normalizar con MinMaxScaler [0, 1]
# (misma estrategia que usan los scripts de los algoritmos del ingeniero)
print("\nEstandarizando todas las características con MinMaxScaler [0, 1]...")
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Reconstruir DataFrame con nombres de columna y variable objetivo
df_ml = pd.DataFrame(X_scaled, columns=feature_names)
df_ml['Diagnostico'] = y.values

print(f"\nMuestra de los datos preprocesados:")
print(df_ml.head())

print(f"\nRango de valores tras escalar:")
print(f"  Mínimo: {df_ml[feature_names].min().min():.4f}")
print(f"  Máximo: {df_ml[feature_names].max().max():.4f}")

# =============================================================================
print("\n--- 5. EXPORTACIÓN DEL DATASET PREPROCESADO ---")
# =============================================================================

output_path = os.path.join(_DIR, 'breast_cancer_preprocessed.csv')
df_ml.to_csv(output_path, index=False)

print(f"\n¡Proceso exitoso! Datos preprocesados guardados en:")
print(f"  '{output_path}'")
print(f"\nDimensiones finales del CSV: {df_ml.shape[0]} filas x {df_ml.shape[1]} columnas")
print(f"  - {df_ml.shape[1]-1} características (escaladas a [0,1])")
print(f"  - 1 variable objetivo 'Diagnostico' (0=Benigno, 1=Maligno)")
