# -*- coding: utf-8 -*-
"""
Comparativa de todos los algoritmos de ML para los 3 datasets:
  - Cancer de Mama
  - Diabetes
  - Titanic

Algoritmos evaluados por dataset:
  1. Regresion Logistica
  2. K-NN
  3. Arbol de Decision
  4. Random Forest

Usa GridSearchCV para optimizar hiperparametros.
Genera: comparativa_algoritmos.xlsx
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, RadarChart, Reference
from openpyxl.chart.series import SeriesLabel

warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# COLUMNAS DEL EXCEL (mismas que en la imagen del ingeniero)
# =============================================================================
METRICAS = ['Accuracy', 'Precision', 'Recall (Sensibilidad)', 'Specificity', 'F1-Score', 'AUC']
ALG_NOMBRES = ['KNN', 'Arbol de Decision', 'Random Forest', 'Regresion Logistica']

# =============================================================================
# HELPER - calcula todas las metricas
# =============================================================================
def calcular_metricas(y_test, y_pred, y_pred_proba=None):
    metricas = {}
    metricas['Accuracy']                = round(accuracy_score(y_test, y_pred), 8)
    metricas['Precision']               = round(precision_score(y_test, y_pred, zero_division=0), 8)
    metricas['Recall (Sensibilidad)']   = round(recall_score(y_test, y_pred, zero_division=0), 8)
    metricas['Specificity']             = round(recall_score(y_test, y_pred, pos_label=0, zero_division=0), 8)
    metricas['F1-Score']                = round(f1_score(y_test, y_pred, zero_division=0), 8)
    try:
        proba = y_pred_proba if y_pred_proba is not None else y_pred
        metricas['AUC'] = round(roc_auc_score(y_test, proba), 8)
    except ValueError:
        metricas['AUC'] = 0.0
    return metricas

# =============================================================================
# GRIDS DE HIPERPARAMETROS PARA GridSearchCV
# =============================================================================
GRID_LR = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [500, 1000]
}

GRID_KNN = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

GRID_DT = {
    'max_depth': [3, 5, 7, 10],
    'criterion': ['gini', 'entropy']
}

GRID_RF = {
    'n_estimators': [10, 15, 50, 100],
    'max_depth': [3, 4, 5, 7],
    'criterion': ['gini', 'entropy']
}

def entrenar_con_grid(model, param_grid, X_train, y_train, cv=5):
    """Entrena un modelo usando GridSearchCV y retorna el mejor modelo."""
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"    Mejores parametros: {grid.best_params_}")
    print(f"    Mejor score CV:     {grid.best_score_:.4f}")
    return grid.best_estimator_

def evaluar_modelo(nombre_alg, model, X_test, y_test):
    """Predice y calcula metricas para un modelo entrenado."""
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = None
    return {
        'Algoritmo': nombre_alg,
        **calcular_metricas(y_test, y_pred, y_proba)
    }

# =============================================================================
# Almacen de resultados por dataset
# =============================================================================
resultados_por_dataset = {}

# =============================================================================
# 1. CANCER DE MAMA
# =============================================================================
print("=" * 60)
print("PROCESANDO: Cancer de Mama (con GridSearchCV)")
print("=" * 60)

dataset_cm = load_breast_cancer()
X_cm = dataset_cm.data
y_cm = 1 - dataset_cm.target  # 1=maligno, 0=benigno

X_train_cm, X_test_cm, y_train_cm, y_test_cm = train_test_split(
    X_cm, y_cm, test_size=0.2, random_state=42, stratify=y_cm)

scaler_cm = MinMaxScaler(feature_range=(0, 1))
X_train_cm_s = scaler_cm.fit_transform(X_train_cm)
X_test_cm_s = scaler_cm.transform(X_test_cm)

res_cm = []

# 1.1 KNN
print("  > KNN...")
m = entrenar_con_grid(KNeighborsClassifier(), GRID_KNN, X_train_cm_s, y_train_cm)
res_cm.append(evaluar_modelo('KNN', m, X_test_cm_s, y_test_cm))

# 1.2 Arbol de Decision
print("  > Arbol de Decision...")
m = entrenar_con_grid(DecisionTreeClassifier(random_state=42), GRID_DT, X_train_cm_s, y_train_cm)
res_cm.append(evaluar_modelo('Arbol de Decision', m, X_test_cm_s, y_test_cm))

# 1.3 Random Forest
print("  > Random Forest...")
m = entrenar_con_grid(RandomForestClassifier(random_state=42), GRID_RF, X_train_cm_s, y_train_cm)
res_cm.append(evaluar_modelo('Random Forest', m, X_test_cm_s, y_test_cm))

# 1.4 Regresion Logistica
print("  > Regresion Logistica...")
m = entrenar_con_grid(LogisticRegression(), GRID_LR, X_train_cm_s, y_train_cm)
res_cm.append(evaluar_modelo('Regresion Logistica', m, X_test_cm_s, y_test_cm))

resultados_por_dataset['Cancer de Mama'] = res_cm

# =============================================================================
# 2. DIABETES
# =============================================================================
print("\n" + "=" * 60)
print("PROCESANDO: Diabetes (con GridSearchCV)")
print("=" * 60)

csv_diabetes = os.path.normpath(
    os.path.join(_DIR, 'Diabetes', '1. Preprocesamiento', 'diabetes_preprocessed.csv'))

try:
    df_diab = pd.read_csv(csv_diabetes)
    X_diab = df_diab.drop(columns=['Hubo_Hipoglucemia']).values
    y_diab = df_diab['Hubo_Hipoglucemia'].values

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab)

    scaler_d = MinMaxScaler(feature_range=(0, 1))
    X_train_d_s = scaler_d.fit_transform(X_train_d)
    X_test_d_s = scaler_d.transform(X_test_d)

    res_d = []

    # Grids con class_weight balanced para diabetes (desbalanceado)
    GRID_LR_BAL = {**GRID_LR, 'class_weight': ['balanced']}
    GRID_DT_BAL = {**GRID_DT, 'class_weight': ['balanced']}
    GRID_RF_BAL = {**GRID_RF, 'class_weight': ['balanced']}

    print("  > KNN...")
    m = entrenar_con_grid(KNeighborsClassifier(), GRID_KNN, X_train_d_s, y_train_d)
    res_d.append(evaluar_modelo('KNN', m, X_test_d_s, y_test_d))

    print("  > Arbol de Decision...")
    m = entrenar_con_grid(DecisionTreeClassifier(random_state=42), GRID_DT_BAL, X_train_d_s, y_train_d)
    res_d.append(evaluar_modelo('Arbol de Decision', m, X_test_d_s, y_test_d))

    print("  > Random Forest...")
    m = entrenar_con_grid(RandomForestClassifier(random_state=42), GRID_RF_BAL, X_train_d_s, y_train_d)
    res_d.append(evaluar_modelo('Random Forest', m, X_test_d_s, y_test_d))

    print("  > Regresion Logistica...")
    m = entrenar_con_grid(LogisticRegression(), GRID_LR_BAL, X_train_d_s, y_train_d)
    res_d.append(evaluar_modelo('Regresion Logistica', m, X_test_d_s, y_test_d))

    resultados_por_dataset['Diabetes'] = res_d

except FileNotFoundError:
    print(f"  [!] No se encontro: {csv_diabetes}")
    resultados_por_dataset['Diabetes'] = []

# =============================================================================
# 3. TITANIC
# =============================================================================
print("\n" + "=" * 60)
print("PROCESANDO: Titanic (con GridSearchCV)")
print("=" * 60)

csv_titanic = os.path.normpath(
    os.path.join(_DIR, 'Titanic', 'Dataset', 'train_procesado_2.csv'))

try:
    df_tit = pd.read_csv(csv_titanic)
    X_tit = df_tit.drop(columns=['Survived']).values
    y_tit = df_tit['Survived'].values

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_tit, y_tit, test_size=0.2, random_state=42, stratify=y_tit)

    res_t = []

    print("  > KNN...")
    m = entrenar_con_grid(KNeighborsClassifier(), GRID_KNN, X_train_t, y_train_t)
    res_t.append(evaluar_modelo('KNN', m, X_test_t, y_test_t))

    print("  > Arbol de Decision...")
    m = entrenar_con_grid(DecisionTreeClassifier(random_state=42), GRID_DT, X_train_t, y_train_t)
    res_t.append(evaluar_modelo('Arbol de Decision', m, X_test_t, y_test_t))

    print("  > Random Forest...")
    m = entrenar_con_grid(RandomForestClassifier(random_state=42), GRID_RF, X_train_t, y_train_t)
    res_t.append(evaluar_modelo('Random Forest', m, X_test_t, y_test_t))

    print("  > Regresion Logistica...")
    m = entrenar_con_grid(LogisticRegression(), {**GRID_LR, 'class_weight': ['balanced']}, X_train_t, y_train_t)
    res_t.append(evaluar_modelo('Regresion Logistica', m, X_test_t, y_test_t))

    resultados_por_dataset['Titanic'] = res_t

except FileNotFoundError:
    print(f"  [!] No se encontro: {csv_titanic}")
    resultados_por_dataset['Titanic'] = []

# =============================================================================
# GENERAR EXCEL CON FORMATO COMO LA IMAGEN DEL INGENIERO
# =============================================================================
print("\n" + "=" * 60)
print("GENERANDO ARCHIVO EXCEL...")
print("=" * 60)

wb = Workbook()
# Borrar hoja por defecto
wb.remove(wb.active)

# Estilos
thin = Side(style='thin', color='000000')
border_all = Border(left=thin, right=thin, top=thin, bottom=thin)

# Colores de encabezado por dataset (similar a la imagen: fondo oscuro, texto claro)
DATASET_STYLES = {
    'Cancer de Mama': {'header_fill': PatternFill('solid', fgColor='4A0E4E'),
                       'header_font': Font(bold=True, size=11, color='FFFFFF', name='Calibri'),
                       'row_fills': [PatternFill('solid', fgColor='E8D5E8'),
                                     PatternFill('solid', fgColor='F5EDF5')]},
    'Diabetes':       {'header_fill': PatternFill('solid', fgColor='1B5E20'),
                       'header_font': Font(bold=True, size=11, color='FFFFFF', name='Calibri'),
                       'row_fills': [PatternFill('solid', fgColor='C8E6C9'),
                                     PatternFill('solid', fgColor='E8F5E9')]},
    'Titanic':        {'header_fill': PatternFill('solid', fgColor='0D47A1'),
                       'header_font': Font(bold=True, size=11, color='FFFFFF', name='Calibri'),
                       'row_fills': [PatternFill('solid', fgColor='BBDEFB'),
                                     PatternFill('solid', fgColor='E3F2FD')]},
}

for ds_name, res_list in resultados_por_dataset.items():
    if not res_list:
        continue

    ws = wb.create_sheet(title=ds_name[:31])  # max 31 chars for sheet name
    styles = DATASET_STYLES.get(ds_name, DATASET_STYLES['Cancer de Mama'])

    # --- Encabezados ---
    headers = ['Algoritmo'] + METRICAS
    col_widths = [20, 14, 14, 22, 14, 14, 14]

    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.fill = styles['header_fill']
        cell.font = styles['header_font']
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border_all
        ws.column_dimensions[get_column_letter(col_idx)].width = col_widths[col_idx - 1]

    ws.row_dimensions[1].height = 25

    # Activar autofiltro (flechitas como en la imagen)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    # --- Datos ---
    for i, res in enumerate(res_list):
        row = i + 2
        bg = styles['row_fills'][i % 2]

        # Algoritmo
        cell = ws.cell(row=row, column=1, value=res['Algoritmo'])
        cell.font = Font(bold=True, size=10, name='Calibri')
        cell.fill = bg
        cell.alignment = Alignment(horizontal='left', vertical='center', indent=1)
        cell.border = border_all

        # Metricas
        for col_idx, metrica in enumerate(METRICAS, start=2):
            val = res.get(metrica, 0.0)
            cell = ws.cell(row=row, column=col_idx, value=val)
            cell.font = Font(size=10, name='Calibri')
            cell.fill = bg
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_all
            if isinstance(val, float):
                cell.number_format = '0.00000000'

        ws.row_dimensions[row].height = 20

    num_rows_data = len(res_list)
    data_end_row = 1 + num_rows_data  # row 1 is header, data starts at row 2

    # --- GRAFICO DE BARRAS (como en la imagen) ---
    chart_bar = BarChart()
    chart_bar.type = "col"
    chart_bar.grouping = "clustered"
    chart_bar.title = "Comparacion"
    chart_bar.y_axis.title = None
    chart_bar.x_axis.title = None
    chart_bar.style = 10
    chart_bar.width = 22
    chart_bar.height = 14

    # Categorias = metricas (eje X)
    cats = Reference(ws, min_col=2, max_col=len(headers), min_row=1, max_row=1)
    chart_bar.set_categories(cats)

    # Colores para las barras de cada algoritmo
    bar_colors = ['1F4E79', 'C00000', '375623', '000000']
    for i in range(num_rows_data):
        values = Reference(ws, min_col=2, max_col=len(headers),
                          min_row=i + 2, max_row=i + 2)
        chart_bar.add_data(values, titles_from_data=False)
        chart_bar.series[i].title = SeriesLabel(v=res_list[i]['Algoritmo'])
        chart_bar.series[i].graphicalProperties.solidFill = bar_colors[i % len(bar_colors)]

    chart_bar.legend.position = 'b'
    ws.add_chart(chart_bar, f"A{data_end_row + 2}")

    # --- GRAFICO DE RADAR (como en la imagen) ---
    chart_radar = RadarChart()
    chart_radar.type = "marker"
    chart_radar.title = "Comparacion"
    chart_radar.style = 26
    chart_radar.width = 16
    chart_radar.height = 14

    cats_radar = Reference(ws, min_col=2, max_col=len(headers), min_row=1, max_row=1)
    chart_radar.set_categories(cats_radar)

    radar_colors = ['1F4E79', 'C00000', '375623', '7030A0']
    for i in range(num_rows_data):
        values = Reference(ws, min_col=2, max_col=len(headers),
                          min_row=i + 2, max_row=i + 2)
        chart_radar.add_data(values, titles_from_data=False)
        chart_radar.series[i].title = SeriesLabel(v=res_list[i]['Algoritmo'])

    chart_radar.legend.position = 'r'
    # Posicionar el radar a la derecha del bar chart
    ws.add_chart(chart_radar, f"I{data_end_row + 2}")

# =============================================================================
# HOJA RESUMEN GENERAL (todos los datasets juntos)
# =============================================================================
ws_resumen = wb.create_sheet(title='Resumen General')

# Titulo
ws_resumen.merge_cells('A1:H1')
c = ws_resumen['A1']
c.value = 'Comparativa General - Todos los Datasets (con GridSearchCV)'
c.font = Font(bold=True, size=14, color='FFFFFF', name='Calibri')
c.fill = PatternFill('solid', fgColor='263238')
c.alignment = Alignment(horizontal='center', vertical='center')
ws_resumen.row_dimensions[1].height = 28

# Encabezados
headers_res = ['Dataset', 'Algoritmo'] + METRICAS
col_widths_res = [18, 20, 14, 14, 22, 14, 14, 14]
for col_idx, h in enumerate(headers_res, start=1):
    cell = ws_resumen.cell(row=2, column=col_idx, value=h)
    cell.fill = PatternFill('solid', fgColor='37474F')
    cell.font = Font(bold=True, size=11, color='FFFFFF', name='Calibri')
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    cell.border = border_all
    ws_resumen.column_dimensions[get_column_letter(col_idx)].width = col_widths_res[col_idx - 1]

ws_resumen.row_dimensions[2].height = 25

fila = 3
ds_colors = {
    'Cancer de Mama': ['E8D5E8', 'F5EDF5'],
    'Diabetes':       ['C8E6C9', 'E8F5E9'],
    'Titanic':        ['BBDEFB', 'E3F2FD'],
}

for ds_name, res_list in resultados_por_dataset.items():
    colors = ds_colors.get(ds_name, ['FFFFFF', 'F5F5F5'])
    for i, res in enumerate(res_list):
        bg = PatternFill('solid', fgColor=colors[i % 2])

        # Dataset
        cell = ws_resumen.cell(row=fila, column=1, value=ds_name)
        cell.font = Font(bold=True, size=10, name='Calibri')
        cell.fill = bg
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border_all

        # Algoritmo
        cell = ws_resumen.cell(row=fila, column=2, value=res['Algoritmo'])
        cell.font = Font(bold=True, size=10, name='Calibri')
        cell.fill = bg
        cell.alignment = Alignment(horizontal='left', vertical='center', indent=1)
        cell.border = border_all

        # Metricas
        for col_idx, metrica in enumerate(METRICAS, start=3):
            val = res.get(metrica, 0.0)
            cell = ws_resumen.cell(row=fila, column=col_idx, value=val)
            cell.font = Font(size=10, name='Calibri')
            cell.fill = bg
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_all
            if isinstance(val, float):
                cell.number_format = '0.00000000'

        ws_resumen.row_dimensions[fila].height = 18
        fila += 1

# =============================================================================
# GUARDAR
# =============================================================================
output_path = os.path.join(_DIR, 'comparativa_algoritmos.xlsx')
wb.save(output_path)

print(f"\n[OK] Archivo Excel generado exitosamente:")
print(f"   {output_path}")
print(f"\nHojas del Excel:")
for ds in resultados_por_dataset:
    print(f"  - '{ds}' (tabla + grafico de barras + grafico de radar)")
print(f"  - 'Resumen General' (todos los datasets juntos)")
print(f"\nMetricas: Accuracy, Precision, Recall (Sensibilidad), Specificity, F1-Score, AUC")
print(f"Optimizacion: GridSearchCV con cv=5")
