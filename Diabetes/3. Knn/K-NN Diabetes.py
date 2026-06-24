# -*- coding: utf-8 -*-
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
try:
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

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- 2. ENTRENAMIENTO DEL MODELO K-NN ---")
# Crea y entrena el modelo
model = KNeighborsClassifier(n_neighbors=3, p=2, weights='uniform')
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

print("--- 3. EVALUACIÓN DEL MODELO ---")
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

print("\n--- 4. EXPLICABILIDAD CON SHAP ---")
# SHAP es pesado, usamos una muestra como en tu código original
sample_size = min(50, len(X_train), len(X_test)) 
X_train_sample = shap.sample(X_train, sample_size)
X_test_sample = shap.sample(X_test, sample_size)

# Función que solo devuelva la probabilidad de la clase 1
predict_fn = lambda x: model.predict_proba(x)[:, 1]

# Le pasamos nuestra nueva función al KernelExplainer
explainer = shap.KernelExplainer(predict_fn, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names)