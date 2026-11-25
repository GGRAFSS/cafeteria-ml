"""
Módulo Unidad 2: Aprendizaje Supervisado
- Regresión: Predice el Consumo_Total_MXN (Lineal y Árboles)
- Clasificación: Predice Satisfaccion_Alta (Logística, KNN, SVM, Árboles)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)


def _preparar_features(df, target_col, is_classification=False):
    """Preprocesa el dataset: limpia NaNs, codifica categóricas y escala."""
    df = df.copy()

    # 1. Limpieza
    if is_classification:
        df = df.dropna(subset=["Satisfaccion_1_al_5"])
        df["Satisfaccion_Alta"] = (df["Satisfaccion_1_al_5"] >= 4).astype(int)
        y = df["Satisfaccion_Alta"].values
        features_to_drop = ["Satisfaccion_1_al_5", "Satisfaccion_Alta", "ID_Cliente"]
    else:
        df = df.dropna(subset=["Consumo_Total_MXN", "Tiempo_Espera_Min", "Tiempo_Servicio_Min"])
        y = df[target_col].values
        features_to_drop = [target_col, "ID_Cliente", "Satisfaccion_1_al_5"]

    X_df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

    # 2. Codificación de variables categóricas
    cat_cols = ["Tipo_Cliente", "Metodo_Pago", "Es_Hora_Pico"]
    for col in cat_cols:
        if col in X_df.columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str).fillna("Nulo"))

    X = X_df.values

    # 3. División Train/Test (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Escalado (Crucial para KNN y SVM)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test


# ============================================================
# 1. REGRESIÓN
# ============================================================
def entrenar_regresion(df):
    try:
        X_train, X_test, y_train, y_test = _preparar_features(
            df, target_col="Consumo_Total_MXN", is_classification=False
        )
    except Exception as e:
        return {"error": f"Error en preparación: {e}"}

    if len(X_train) < 20:
        return {"error": "Datos insuficientes para entrenar (mínimo 20 registros válidos)."}

    # Modelo 1: Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_lr_train = lr.predict(X_train)   # diagnóstico de over/underfitting

    # Modelo 2: Árbol de Regresión
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_pred_dt_train = dt.predict(X_train)   # diagnóstico de over/underfitting

    # Gráfica: Real vs Predicho (Regresión Lineal)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred_lr, alpha=0.6, color="#4C72B0", edgecolor="white")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Consumo Real (MXN)")
    ax.set_ylabel("Consumo Predicho (MXN)")
    ax.set_title("Regresión Lineal: Real vs Predicho")
    ax.grid(alpha=0.3)

    return {
        "metricas_lr": {"MSE": mean_squared_error(y_test, y_pred_lr), "R2": r2_score(y_test, y_pred_lr)},
        "metricas_dt": {"MSE": mean_squared_error(y_test, y_pred_dt), "R2": r2_score(y_test, y_pred_dt)},
        "metricas_lr_train": {"MSE": mean_squared_error(y_train, y_pred_lr_train),
                              "R2": r2_score(y_train, y_pred_lr_train)},
        "metricas_dt_train": {"MSE": mean_squared_error(y_train, y_pred_dt_train),
                              "R2": r2_score(y_train, y_pred_dt_train)},
        "fig": fig,
        "error": None
    }


# ============================================================
# 2. CLASIFICACIÓN
# ============================================================
def entrenar_clasificacion(df, modelo_nombre):
    try:
        X_train, X_test, y_train, y_test = _preparar_features(
            df, target_col="Satisfaccion_1_al_5", is_classification=True
        )
    except Exception as e:
        return {"error": f"Error en preparación: {e}"}

    if len(X_train) < 20 or len(np.unique(y_train)) < 2:
        return {"error": "Datos insuficientes o solo hay una clase en Satisfacción."}

    # Selección de modelo
    if modelo_nombre == "Regresión Logística":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif modelo_nombre == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif modelo_nombre == "SVM":
        # sklearn >= 1.9 deprecó SVC(probability=True).
        # CalibratedClassifierCV provee predict_proba sin warning y es más estable.
        model = CalibratedClassifierCV(
            SVC(random_state=42), method="sigmoid", cv=3
        )
    else:
        model = DecisionTreeClassifier(max_depth=5, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    # Mismas predicciones sobre entrenamiento (diagnóstico de over/underfitting)
    y_pred_train = model.predict(X_train)
    y_proba_train = (
        model.predict_proba(X_train)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    cm = confusion_matrix(y_test, y_pred)
    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T

    # Figura 1: Matriz de Confusión
    fig_cm, ax1 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Baja (0)", "Alta (1)"],
                yticklabels=["Baja (0)", "Alta (1)"])
    ax1.set_xlabel("Predicho")
    ax1.set_ylabel("Real")
    ax1.set_title(f"Matriz de Confusión — {modelo_nombre}")

    # Figura 2: Curva ROC
    fig_roc = None
    auc_val = None
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        fig_roc, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
        ax2.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        ax2.set_xlabel("Falsos Positivos (FPR)")
        ax2.set_ylabel("Verdaderos Positivos (TPR)")
        ax2.set_title("Curva ROC")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)

    # AUC sobre entrenamiento (diagnóstico de over/underfitting)
    auc_train = None
    if y_proba_train is not None:
        fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
        auc_train = auc(fpr_train, tpr_train)

    return {
        "metricas": {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "AUC": auc_val,
        },
        "metricas_train": {
            "Accuracy": accuracy_score(y_train, y_pred_train),
            "Precision": precision_score(y_train, y_pred_train, zero_division=0),
            "Recall": recall_score(y_train, y_pred_train, zero_division=0),
            "F1": f1_score(y_train, y_pred_train, zero_division=0),
            "AUC": auc_train,
        },
        "fig_cm": fig_cm,
        "fig_roc": fig_roc,
        "reporte": report,
        "error": None,
    }