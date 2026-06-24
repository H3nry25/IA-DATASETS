# -*- coding: utf-8 -*-
"""
@author: IVAN
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########## Configuración de Rutas Absolutas ##########

directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_dataset = os.path.join(directorio_actual, '..', 'dataset')
ruta_archivo = os.path.join(ruta_dataset, 'titanic.csv')

########## Importando y Limpiando la data ##########

########## Importando y Limpiando la data ##########

titanic_df = pd.read_csv(ruta_archivo, sep=';')

# 1. Forzamos la columna Age a ser numérica (el '20%' se vuelve nulo)
titanic_df['Age'] = pd.to_numeric(titanic_df['Age'], errors='coerce')

# 2. ¡NUEVO! Limpiamos las edades imposibles (mayores a 100 años las volvemos nulas)
titanic_df.loc[titanic_df['Age'] > 100, 'Age'] = np.nan

# 3. ¡NUEVO! Filtramos para quedarnos SOLO con las clases de ticket válidas (1, 2 y 3)
titanic_df = titanic_df[titanic_df['Pclass'].isin([1, 2, 3])]

# Ver las primeras filas del conjunto de datos
print("Primeras filas del dataset:")
print(titanic_df.head(10))

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(titanic_df.describe())

# Estadísticas de variables categóricas
print("\nEstadísticas de variables categóricas:")
print(titanic_df.describe(include=['object']))

########## Visualizaciones ##########

# Histogramas de Edades (frecuencias):
plt.figure(figsize=(8, 6))
sns.histplot(titanic_df['Age'].dropna(), bins=30, kde=True)
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.title('Distribución de Edades en el Titanic (Corregido)')
plt.show()

# Crear el diagrama de barras de las clases del target (Survived)
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Distribución de sobrevivientes')
plt.xticks([0, 1], ['No', 'Si'])
plt.show()

print('\nConteo de valores del target:\n', titanic_df['Survived'].value_counts())

# Gráfico de Barras de Supervivencia por Clase:
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Pclass', hue='Survived')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.title('Supervivencia por Clase en el Titanic')
plt.legend(title='Sobreviviente', labels=['No', 'Sí'])
plt.show()

# Gráfico de Torta de Género:
plt.figure(figsize=(8, 6))
titanic_df['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribución de Género en el Titanic')
plt.ylabel('')
plt.show()

########## Análisis de Correlación ##########

# Seleccionar las variables numéricas principales para el análisis de correlación y el target
variables_numericas = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived']

# Crear una submatriz de correlación
correlation_matrix = titanic_df[variables_numericas].corr()

# crea una máscara para ocultar la parte superior de la matriz de correlación
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Crear un mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación entre Variables Numéricas del Titanic')
plt.show()

# Aplicar una máscara para mostrar solo correlaciones moderadas/altas mayores a 0.4
mask_moderadas = np.abs(correlation_matrix) < 0.4
correlation_matrix[mask_moderadas] = np.nan

# Crear un mapa de calor de correlación con valores significativos
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask_moderadas, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación (moderadas / altas)')
plt.show()

########## Detección y manejo de Atípicos ##########

# Crear el box-plot de la variable "edad"
plt.figure(figsize=(10, 6))
sns.boxplot(x=titanic_df['Age'])
plt.title('Box-Plot de la Edad')
plt.xlabel('Edad')
plt.show()

# Eliminar valores atípicos (filas)
Q1 = titanic_df['Age'].quantile(0.25)
Q3 = titanic_df['Age'].quantile(0.75)
IQR = Q3 - Q1 # rango intercuartil
limite_superior = Q3 + 1.5*IQR
limite_inferior = Q1 - 1.5*IQR
filtered_titanic_df = titanic_df[(titanic_df['Age'] >= limite_inferior) & (titanic_df['Age'] <= limite_superior)]

# Crear el box-plot de la variable "edad" sin atípicos
plt.figure(figsize=(10, 6))
sns.boxplot(x=filtered_titanic_df['Age'])
plt.title('Box-Plot de la Edad (sin atípicos)')
plt.xlabel('Edad')
plt.show()

print("\n¡Análisis Exploratorio terminado con éxito! Revisa las ventanas de los gráficos.")