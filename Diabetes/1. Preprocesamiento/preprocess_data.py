import os
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("--- 1. LECTURA DE DATOS ---")
path = r'C:\Users\SUPERTRONICA\Documents\GitHub\IA-DATASETS\Diabetes\Dataset'
all_files = glob.glob(os.path.join(path, "data-*"))

list_df = []
for filename in all_files:
    df_temp = pd.read_csv(filename, sep='\t', header=None, 
                          names=['Fecha', 'Hora', 'Codigo', 'Valor'], 
                          on_bad_lines='skip')
    df_temp['Paciente'] = os.path.basename(filename)
    list_df.append(df_temp)

df_completo = pd.concat(list_df, ignore_index=True)

print("--- 2. LIMPIEZA INICIAL ---")
print("Borrando columna 'Hora' (no requerida para análisis diario)...")
df_completo = df_completo.drop(columns=['Hora'])

df_completo['Valor'] = pd.to_numeric(df_completo['Valor'], errors='coerce')
df_completo = df_completo.dropna(subset=['Valor'])

print("--- 3. INGENIERÍA DE CARACTERÍSTICAS ---")
df_target = df_completo[df_completo['Codigo'] == 65].groupby(['Paciente', 'Fecha']).size().reset_index(name='Hubo_Hipoglucemia')
df_target['Hubo_Hipoglucemia'] = 1 

df_features = df_completo[df_completo['Codigo'].isin([33, 34, 58, 60, 62])].copy()
codigo_map = {33: 'Insulina_Regular', 34: 'Insulina_NPH', 58: 'Glucosa_Desayuno', 
              60: 'Glucosa_Almuerzo', 62: 'Glucosa_Cena'}
df_features['Variable'] = df_features['Codigo'].map(codigo_map)

df_pivot = df_features.pivot_table(index=['Paciente', 'Fecha'], columns='Variable', values='Valor', aggfunc='mean').reset_index()
# Llenar faltantes con la mediana
df_pivot = df_pivot.fillna(df_pivot.median(numeric_only=True))

df_final = pd.merge(df_pivot, df_target, on=['Paciente', 'Fecha'], how='left')
df_final['Hubo_Hipoglucemia'] = df_final['Hubo_Hipoglucemia'].fillna(0).astype(int)

print("--- 4. PREPROCESAMIENTO FINAL Y ESTANDARIZACIÓN ---")
print("Borrando columnas 'Paciente' y 'Fecha'...")
df_ml = df_final.drop(columns=['Paciente', 'Fecha'])

feature_names = ['Glucosa_Almuerzo', 'Glucosa_Cena', 'Glucosa_Desayuno', 'Insulina_NPH', 'Insulina_Regular']

print("Estandarizando columnas con MinMaxScaler...")
scaler = MinMaxScaler(feature_range=(0,1))
# Se estandarizan todas las features (se sobreescriben en el dataframe)
df_ml[feature_names] = scaler.fit_transform(df_ml[feature_names])

# Guardar el dataset procesado
output_path = 'diabetes_preprocessed.csv'
df_ml.to_csv(output_path, index=False)

print(f"\n¡Proceso exitoso! Datos preprocesados y guardados en '{output_path}'.")
print("\nMuestra de los primeros datos preprocesados:")
print(df_ml.head())
