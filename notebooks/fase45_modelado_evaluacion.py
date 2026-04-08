#!/usr/bin/env python3
"""
Fase IV-V: Modelado y Evaluación
=================================
Entrena, optimiza y evalúa modelos de clasificación para predecir task_success.
Exporta el pipeline final y metadata para el dashboard interactivo.

Uso: python3 fase45_modelado_evaluacion.py
(ejecutar desde dentro de la carpeta notebooks/)
"""

# --- Imports estándar ---
import json
import os
import shutil

# --- Ciencia de datos ---
import joblib
import matplotlib.patches as mpatches

# --- Visualización ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- scikit-learn ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# Constantes globales
# ============================================================

BASE_FASE3 = "results/fase3-preparacion"
OUTPUT_DIR = "results/fase45-modelado"
DASHBOARD_MODEL = "../dashboard/model"
DASHBOARD_ASSETS = "../dashboard/assets"

# Orden exacto de features (igual que en fase3)
FEATURE_NAMES = [
    "hours_coding",
    "coffee_intake_mg",
    "distractions",
    "sleep_hours",
    "commits",
    "bugs_reported",
    "ai_usage_hours",
    "cognitive_load",
    "sleep_deficit",
    "productivity_ratio",
    "caffeine_per_hour",
    "work_intensity",
    "coffee_category",
    "sleep_category",
]

# Etiquetas en español para visualizaciones
FEATURE_LABELS = {
    "hours_coding": "Horas de Código",
    "coffee_intake_mg": "Cafeína (mg)",
    "distractions": "Distracciones",
    "sleep_hours": "Horas de Sueño",
    "commits": "Commits",
    "bugs_reported": "Bugs Reportados",
    "ai_usage_hours": "Uso de IA (h)",
    "cognitive_load": "Carga Cognitiva",
    "sleep_deficit": "Déficit de Sueño",
    "productivity_ratio": "Ratio Productividad",
    "caffeine_per_hour": "Cafeína/Hora",
    "work_intensity": "Intensidad Trabajo",
    "coffee_category": "Categoría Cafeína",
    "sleep_category": "Categoría Sueño",
}

# Paleta de colores consistente
COLOR_TEAL = "#4ECDC4"
COLOR_RED = "#FF6B6B"
COLOR_YELLOW = "#FFE66D"
COLOR_GRAY = "#b2bec3"
COLOR_GREEN = "#00b894"
COLOR_BLUE = "#0984e3"
COLOR_ORANGE = "#e17055"


# ============================================================
# Helpers
# ============================================================


def _classify_prediction(true_val, pred_val):
    """Clasifica cada predicción en TP, TN, FP o FN."""
    if true_val == 1 and pred_val == 1:
        return "TP"
    elif true_val == 0 and pred_val == 0:
        return "TN"
    elif true_val == 0 and pred_val == 1:
        return "FP"
    else:
        return "FN"


def _print_separator(char="=", width=65):
    print(char * width)


# ============================================================
# Paso 1 — Cargar datos preparados por fase3
# ============================================================


def load_data():
    """Carga los conjuntos de datos generados por el script de Fase 3."""
    print("\n📂 Paso 1: Cargando datos de Fase 3...")

    # Verificar que los archivos necesarios existen
    required = [
        "X_train.csv",
        "X_val.csv",
        "X_test.csv",
        "y_train.csv",
        "y_val.csv",
        "y_test.csv",
        "preprocessing_info.json",
    ]
    for fname in required:
        path = f"{BASE_FASE3}/{fname}"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Archivo requerido no encontrado: {path}\n"
                "  → Ejecutá primero el script de Fase 3 para generar los splits."
            )

    X_train = pd.read_csv(f"{BASE_FASE3}/X_train.csv")
    X_val = pd.read_csv(f"{BASE_FASE3}/X_val.csv")
    X_test = pd.read_csv(f"{BASE_FASE3}/X_test.csv")
    y_train = pd.read_csv(f"{BASE_FASE3}/y_train.csv").squeeze()
    y_val = pd.read_csv(f"{BASE_FASE3}/y_val.csv").squeeze()
    y_test = pd.read_csv(f"{BASE_FASE3}/y_test.csv").squeeze()

    with open(f"{BASE_FASE3}/preprocessing_info.json", encoding="utf-8") as f:
        prep_info = json.load(f)

    print(
        f"   X_train : {X_train.shape}  |  X_val : {X_val.shape}  |  X_test : {X_test.shape}"
    )
    print(
        f"   y_train : {len(y_train)}        |  y_val : {len(y_val)}       |  y_test : {len(y_test)}"
    )
    print(
        f"   Features en preprocessing_info: {prep_info.get('n_features', len(FEATURE_NAMES))}"
    )
    print("   ✅ Datos cargados correctamente")

    return X_train, X_val, X_test, y_train, y_val, y_test, prep_info


# ============================================================
# Paso 2 — Escalar features
# ============================================================


def scale_features(X_train, X_val, X_test):
    """Ajusta StandardScaler en train y transforma todos los splits."""
    print("\n⚙️  Paso 2: Escalando features con StandardScaler...")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    print("   ✅ Escalado completado (fit en train, transform en val/test)")
    return scaler, X_train_sc, X_val_sc, X_test_sc


# ============================================================
# Paso 3 — Entrenar modelos baseline
# ============================================================


def train_baselines(X_train_sc, y_train):
    """Entrena los 3 modelos baseline y devuelve un dict con los objetos."""
    print("\n🤖 Paso 3: Entrenando modelos baseline...")

    # Modelo 1: Logistic Regression
    print("   📌 Entrenando Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    print("   ✅ Logistic Regression listo")

    # Modelo 2: Decision Tree
    print("   📌 Entrenando Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_sc, y_train)
    print("   ✅ Decision Tree listo")

    # Modelo 3: Random Forest
    print("   📌 Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_sc, y_train)
    print("   ✅ Random Forest listo")

    return {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
    }


# ============================================================
# Paso 4 — Evaluar modelos en validation set
# ============================================================


def evaluate_on_val(models, X_val_sc, y_val):
    """Calcula métricas de cada modelo en el validation set y muestra tabla."""
    print("\n📊 Paso 4: Evaluando modelos en Validation Set...")

    val_metrics = {}
    for name, model in models.items():
        y_pred_val = model.predict(X_val_sc)
        y_prob_val = model.predict_proba(X_val_sc)[:, 1]
        val_metrics[name] = {
            "accuracy": accuracy_score(y_val, y_pred_val),
            "precision": precision_score(y_val, y_pred_val, zero_division=0),
            "recall": recall_score(y_val, y_pred_val, zero_division=0),
            "f1": f1_score(y_val, y_pred_val, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob_val),
        }

    # Tabla comparativa
    print()
    _print_separator()
    print(
        f"{'Modelo':<24} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    )
    _print_separator("-")
    for name, m in val_metrics.items():
        print(
            f"{name:<24} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )
    _print_separator()

    return val_metrics


# ============================================================
# Paso 5 — Optimización del mejor modelo (Random Forest)
# ============================================================


def optimize_random_forest(X_train_sc, y_train):
    """GridSearchCV sobre Random Forest. Devuelve el mejor estimador."""
    print("\n🔧 Paso 5: Optimizando Random Forest con GridSearchCV...")
    print("   (Esto puede tardar varios minutos con n_jobs=-1)")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train_sc, y_train)
    best_model = grid_search.best_estimator_

    print(f"   ✅ GridSearchCV completado")
    print(f"   Mejor F1 (CV interno): {grid_search.best_score_:.4f}")
    print(f"   Mejores parámetros:    {grid_search.best_params_}")

    return grid_search, best_model


# ============================================================
# Paso 6 — Evaluación final en TEST SET
# ============================================================


def evaluate_on_test(
    best_model, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test
):
    """Evalúa el modelo final en el test set (nunca visto)."""
    print("\n🎯 Paso 6: Evaluación final en Test Set (nunca visto)...")

    y_pred = best_model.predict(X_test_sc)
    y_prob = best_model.predict_proba(X_test_sc)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print()
    _print_separator()
    print("  TEST SET — Random Forest Optimizado")
    _print_separator("-")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    _print_separator()

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Exitoso", "Exitoso"]))

    print("🧩 Confusion Matrix:")
    print(cm)

    # Cross-validation sobre train+val (datos ya escalados)
    X_trainval_sc = np.vstack([X_train_sc, X_val_sc])
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)
    cv_scores = cross_val_score(
        best_model, X_trainval_sc, y_trainval, cv=5, scoring="f1"
    )
    print(
        f"\n📈 Cross-Validation (5-fold, train+val) — F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

    test_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    return y_pred, y_prob, cm, test_metrics, cv_scores


# ============================================================
# Paso 7 — Visualizaciones
# ============================================================


def plot_model_comparison(val_metrics):
    """Barplot comparativo de los 3 modelos en validation set."""
    print("   📊 Generando model_comparison...")

    model_names = list(val_metrics.keys())
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    model_colors = [COLOR_TEAL, COLOR_RED, COLOR_YELLOW]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Comparativa de Modelos — Validation Set",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    # --- Subplot izquierdo: Accuracy vs F1 por modelo ---
    x = np.arange(len(model_names))
    width = 0.35
    accs = [val_metrics[m]["accuracy"] for m in model_names]
    f1s = [val_metrics[m]["f1"] for m in model_names]

    bars1 = axes[0].bar(
        x - width / 2,
        accs,
        width,
        label="Accuracy",
        color=COLOR_TEAL,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = axes[0].bar(
        x + width / 2,
        f1s,
        width,
        label="F1-Score",
        color=COLOR_RED,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar in bars1:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.008,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in bars2:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.008,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axes[0].set_title("Accuracy vs F1-Score", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_axisbelow(True)

    # --- Subplot derecho: todas las métricas ---
    x2 = np.arange(len(metrics_keys))
    width2 = 0.25

    for i, (name, color) in enumerate(zip(model_names, model_colors)):
        vals = [val_metrics[name][mk] for mk in metrics_keys]
        bars = axes[1].bar(
            x2 + (i - 1) * width2,
            vals,
            width2,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
        )

    axes[1].set_title("Todas las Métricas", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metrics_labels, fontsize=10)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fase45_model_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ fase45_model_comparison.png")
    return out


def plot_confusion_matrix(cm, accuracy):
    """Heatmap de la confusion matrix con celdas verdes (correctas) y rojas (errores)."""
    print("   📊 Generando confusion_matrix...")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Verde para diagonal (TP, TN), rojo para off-diagonal (FP, FN)
    bg_colors = [["#d4edda", "#f8d7da"], ["#f8d7da", "#d4edda"]]
    cell_label = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(8, 7))

    for i in range(2):
        for j in range(2):
            rect = mpatches.FancyBboxPatch(
                (j - 0.5, i - 0.5),
                1,
                1,
                boxstyle="square,pad=0",
                facecolor=bg_colors[i][j],
                edgecolor="white",
                linewidth=3,
            )
            ax.add_patch(rect)

            # Valor absoluto (grande)
            ax.text(
                j,
                i - 0.13,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=26,
                fontweight="bold",
                color="#2d3436",
            )
            # Porcentaje
            ax.text(
                j,
                i + 0.18,
                f"({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                fontsize=13,
                color="#636e72",
            )
            # Etiqueta de tipo
            ax.text(
                j,
                i + 0.38,
                cell_label[i][j],
                ha="center",
                va="center",
                fontsize=11,
                color="#636e72",
                style="italic",
            )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Exitoso", "Pred: Exitoso"], fontsize=11)
    ax.set_yticklabels(["Real: No Exitoso", "Real: Exitoso"], fontsize=11)
    ax.set_xlabel("Predicción", fontsize=12, labelpad=10)
    ax.set_ylabel("Valor Real", fontsize=12, labelpad=10)
    ax.set_title(
        f"Confusion Matrix — Random Forest Optimizado\nAccuracy: {accuracy:.1%}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    legend_handles = [
        mpatches.Patch(
            facecolor="#d4edda", edgecolor="grey", label="Correctos (TP / TN)"
        ),
        mpatches.Patch(
            facecolor="#f8d7da", edgecolor="grey", label="Errores  (FP / FN)"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fase45_confusion_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ fase45_confusion_matrix.png")
    return out


def plot_roc_curve(y_test, y_prob, roc_auc):
    """Curva ROC con AUC, línea de referencia y punto de corte óptimo marcado."""
    print("   📊 Generando roc_curve...")

    fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_test, y_prob)

    # Punto de corte óptimo mediante el índice de Youden (max TPR - FPR)
    optimal_idx = int(np.argmax(tpr_arr - fpr_arr))
    optimal_threshold = float(thresholds_arr[optimal_idx])
    optimal_fpr = float(fpr_arr[optimal_idx])
    optimal_tpr = float(tpr_arr[optimal_idx])

    fig, ax = plt.subplots(figsize=(9, 7))

    # Área bajo la curva (relleno sutil)
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.07, color=COLOR_TEAL)

    # Curva principal
    ax.plot(
        fpr_arr,
        tpr_arr,
        color=COLOR_TEAL,
        lw=2.5,
        label=f"Random Forest Optimizado (AUC = {roc_auc:.4f})",
    )

    # Línea diagonal de referencia (clasificador aleatorio)
    ax.plot(
        [0, 1],
        [0, 1],
        color=COLOR_GRAY,
        linestyle="--",
        lw=1.5,
        label="Modelo aleatorio (AUC = 0.5000)",
    )

    # Punto de corte óptimo
    ax.scatter(
        [optimal_fpr],
        [optimal_tpr],
        color=COLOR_RED,
        s=130,
        zorder=6,
        label=f"Punto óptimo (umbral = {optimal_threshold:.3f})",
    )

    # Anotación del punto óptimo (ajuste dinámico para evitar salirse del gráfico)
    offset_x = 0.12 if optimal_fpr < 0.6 else -0.25
    offset_y = -0.10
    ax.annotate(
        f"({optimal_fpr:.2f}, {optimal_tpr:.2f})",
        xy=(optimal_fpr, optimal_tpr),
        xytext=(optimal_fpr + offset_x, optimal_tpr + offset_y),
        fontsize=10,
        color="#d63031",
        arrowprops=dict(arrowstyle="->", color="#d63031", lw=1.5),
    )

    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title(
        "Curva ROC — Random Forest Optimizado", fontsize=14, fontweight="bold", pad=12
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fase45_roc_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ fase45_roc_curve.png")
    return out, optimal_threshold, optimal_fpr, optimal_tpr


def plot_feature_importance(best_model, accuracy):
    """
    Barplot horizontal de feature importance.
    Top 3 resaltadas en rojo, resto en teal.
    Retorna feat_imp_df y top3_features para uso posterior.
    """
    print("   📊 Generando feature_importance...")

    importances = best_model.feature_importances_
    feat_imp_df = (
        pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "label": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    top3_features = feat_imp_df.nlargest(3, "importance")["feature"].tolist()

    bar_colors = [
        COLOR_RED if feat in top3_features else COLOR_TEAL
        for feat in feat_imp_df["feature"]
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(
        feat_imp_df["label"],
        feat_imp_df["importance"],
        color=bar_colors,
        edgecolor="white",
        linewidth=1.2,
    )

    # Valores al final de cada barra
    for bar, val in zip(bars, feat_imp_df["importance"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#2d3436",
        )

    ax.set_xlabel("Importancia (Gini Impurity)", fontsize=12)
    ax.set_title(
        f"Feature Importance — Random Forest Optimizado\n"
        f"Accuracy en Test: {accuracy:.1%}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(
            facecolor=COLOR_RED, edgecolor="grey", label="Top 3 más importantes"
        ),
        mpatches.Patch(
            facecolor=COLOR_TEAL, edgecolor="grey", label="Resto de features"
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    # Pequeño margen derecho para los valores
    ax.set_xlim(right=feat_imp_df["importance"].max() * 1.18)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fase45_feature_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ fase45_feature_importance.png")
    return out, feat_imp_df, top3_features


def plot_error_analysis(X_test, y_test, y_pred, feat_imp_df):
    """
    Scatter de las 2 features más importantes coloreado por tipo de error
    (TP, TN, FP, FN) en el test set.
    """
    print("   📊 Generando error_analysis...")

    # Top 2 features por importancia
    top2 = feat_imp_df.nlargest(2, "importance")["feature"].tolist()
    feat1, feat2 = top2[0], top2[1]
    label1 = FEATURE_LABELS[feat1]
    label2 = FEATURE_LABELS[feat2]

    X_test_r = X_test.copy().reset_index(drop=True)
    y_test_r = y_test.reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)

    error_types = [_classify_prediction(t, p) for t, p in zip(y_test_r, y_pred_s)]

    error_df = pd.DataFrame(
        {
            "feat1": X_test_r[feat1],
            "feat2": X_test_r[feat2],
            "error_type": error_types,
        }
    )

    # Paleta y marcadores por tipo
    style_map = {
        "TP": {
            "color": COLOR_GREEN,
            "marker": "o",
            "alpha": 0.75,
            "label_prefix": "✅ TP",
        },
        "TN": {
            "color": COLOR_BLUE,
            "marker": "s",
            "alpha": 0.65,
            "label_prefix": "✅ TN",
        },
        "FP": {
            "color": COLOR_RED,
            "marker": "^",
            "alpha": 0.85,
            "label_prefix": "❌ FP",
        },
        "FN": {
            "color": COLOR_ORANGE,
            "marker": "D",
            "alpha": 0.85,
            "label_prefix": "❌ FN",
        },
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for etype in ["TN", "TP", "FN", "FP"]:
        subset = error_df[error_df["error_type"] == etype]
        if len(subset) == 0:
            continue
        s = style_map[etype]
        ax.scatter(
            subset["feat1"],
            subset["feat2"],
            c=s["color"],
            marker=s["marker"],
            s=65,
            alpha=s["alpha"],
            label=f"{s['label_prefix']}  (n={len(subset)})",
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel(label1, fontsize=12)
    ax.set_ylabel(label2, fontsize=12)
    ax.set_title(
        f"Análisis de Errores — Top 2 Features\n"
        f"Verde/Azul = Correcto  ·  Rojo/Naranja = Error",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        title="Tipo de Predicción", fontsize=10, title_fontsize=10, framealpha=0.9
    )
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/fase45_error_analysis.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ fase45_error_analysis.png")
    return out, error_types


def generate_visualizations(
    val_metrics, cm, y_test, y_pred, y_prob, best_model, test_metrics
):
    """Orquesta la generación de todas las visualizaciones del paso 7."""
    print("\n🎨 Paso 7: Generando visualizaciones...")
    plt.style.use("default")
    sns.set_palette("husl")

    plot_model_comparison(val_metrics)

    plot_confusion_matrix(cm, test_metrics["accuracy"])

    _, optimal_threshold, optimal_fpr, optimal_tpr = plot_roc_curve(
        y_test, y_prob, test_metrics["roc_auc"]
    )

    _, feat_imp_df, top3_features = plot_feature_importance(
        best_model, test_metrics["accuracy"]
    )

    _, error_types = plot_error_analysis(
        # X_test sin escalar — para que los ejes tengan las unidades originales
        _GLOBAL["X_test"],
        y_test,
        y_pred,
        feat_imp_df,
    )

    print("   ✅ Todas las visualizaciones generadas")
    return (
        feat_imp_df,
        top3_features,
        error_types,
        optimal_threshold,
        optimal_fpr,
        optimal_tpr,
    )


# ============================================================
# Paso 8 — Guardar modelo y metadata para el dashboard
# ============================================================


def save_pipeline_and_metadata(
    best_model, X_train, X_val, y_train, y_val, test_metrics
):
    """
    Construye un Pipeline (scaler + modelo) entrenado sobre train+val sin escalar,
    lo serializa junto con los feature_names y la metadata del modelo.
    """
    print("\n💾 Paso 8: Guardando pipeline y metadata para el dashboard...")

    # Pipeline: el scaler está DENTRO, por eso se entrena sobre datos crudos
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", best_model.__class__(**best_model.get_params())),
        ]
    )

    X_trainval = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)
    pipeline.fit(X_trainval, y_trainval)

    # 1. Pipeline completo
    pipeline_path = f"{DASHBOARD_MODEL}/pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"   ✅ Pipeline guardado       → {pipeline_path}")

    # 2. Feature names en orden exacto
    feature_path = f"{DASHBOARD_MODEL}/feature_names.pkl"
    joblib.dump(FEATURE_NAMES, feature_path)
    print(f"   ✅ Feature names guardados → {feature_path}")

    # 3. Metadata JSON — serializar manualmente para evitar tipos numpy
    metadata = {
        "model_type": "RandomForestClassifier",
        "best_params": {
            k: (
                None
                if v is None
                else (
                    int(v)
                    if isinstance(v, (bool, int))
                    else float(v)
                    if isinstance(v, float)
                    else v
                )
            )
            for k, v in best_model.get_params().items()
        },
        "test_accuracy": float(round(test_metrics["accuracy"], 4)),
        "test_precision": float(round(test_metrics["precision"], 4)),
        "test_recall": float(round(test_metrics["recall"], 4)),
        "test_f1": float(round(test_metrics["f1"], 4)),
        "test_roc_auc": float(round(test_metrics["roc_auc"], 4)),
        "feature_names": FEATURE_NAMES,
        "feature_importance": dict(
            zip(
                FEATURE_NAMES,
                [float(v) for v in best_model.feature_importances_.tolist()],
            )
        ),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(_GLOBAL["X_test"])),
    }

    meta_path = f"{DASHBOARD_MODEL}/model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Metadata guardada       → {meta_path}")

    return pipeline


# ============================================================
# Paso 9 — Copiar visualizaciones al dashboard/assets
# ============================================================


def copy_to_dashboard():
    """Copia las 4 imágenes principales al directorio de assets del dashboard."""
    print("\n📤 Paso 9: Copiando visualizaciones al dashboard...")

    images = [
        "fase45_feature_importance.png",
        "fase45_confusion_matrix.png",
        "fase45_roc_curve.png",
        "fase45_model_comparison.png",
    ]
    for img in images:
        src = f"{OUTPUT_DIR}/{img}"
        dst = f"{DASHBOARD_ASSETS}/{img}"
        shutil.copy(src, dst)
        print(f"   ✅ {img}")

    print(f"   → Destino: {DASHBOARD_ASSETS}/")


# ============================================================
# Paso 10 — Archivo de estadísticas completo
# ============================================================


def write_statistics(
    val_metrics,
    grid_search,
    best_model,
    test_metrics,
    cv_scores,
    cm,
    feat_imp_df,
    error_types,
    optimal_threshold,
    optimal_fpr,
    optimal_tpr,
):
    """Escribe el archivo de estadísticas con 8 secciones."""
    print("\n📝 Paso 10: Guardando estadísticas completas...")

    total_test = int(len(_GLOBAL["y_test"]))
    error_count = pd.Series(error_types).value_counts()
    tp_count = int(error_count.get("TP", 0))
    tn_count = int(error_count.get("TN", 0))
    fp_count = int(error_count.get("FP", 0))
    fn_count = int(error_count.get("FN", 0))

    top5 = feat_imp_df.nlargest(5, "importance")[["feature", "label", "importance"]]

    accuracy = test_metrics["accuracy"]
    precision = test_metrics["precision"]
    recall = test_metrics["recall"]
    f1 = test_metrics["f1"]
    roc_auc = test_metrics["roc_auc"]

    sep = "=" * 62

    with open(f"{OUTPUT_DIR}/fase45_estadisticas.txt", "w", encoding="utf-8") as f:
        # 1 — Encabezado general
        f.write(f"{sep}\n")
        f.write("=== FASE IV-V: MODELADO Y EVALUACIÓN ===\n")
        f.write(f"{sep}\n")
        f.write("Objetivo : Entrenar y evaluar modelos de clasificación para\n")
        f.write(
            "           predecir task_success a partir de métricas de productividad.\n"
        )
        f.write(
            f"Dataset  : {int(len(_GLOBAL['X_train']))} train + "
            f"{int(len(_GLOBAL['X_val']))} val + {total_test} test\n"
        )
        f.write(f"Features : {len(FEATURE_NAMES)}\n")
        f.write(
            "Modelos evaluados : Logistic Regression, Decision Tree, Random Forest\n\n"
        )

        # 2 — Comparativa de modelos
        f.write(f"{sep}\n")
        f.write("=== COMPARATIVA DE MODELOS (VAL SET) ===\n")
        f.write(f"{sep}\n")
        f.write(
            f"{'Modelo':<24} {'Accuracy':>9} {'Precision':>10} "
            f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9}\n"
        )
        f.write("-" * 72 + "\n")
        for name, m in val_metrics.items():
            f.write(
                f"{name:<24} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}\n"
            )
        f.write("\n")

        # 3 — Modelo final: parámetros óptimos
        f.write(f"{sep}\n")
        f.write("=== MODELO FINAL: RANDOM FOREST OPTIMIZADO ===\n")
        f.write(f"{sep}\n")
        f.write(f"Búsqueda      : GridSearchCV (cv=5, scoring='f1')\n")
        f.write(f"Mejor F1 (CV) : {grid_search.best_score_:.4f}\n\n")
        f.write("Mejores hiperparámetros encontrados:\n")
        for k, v in grid_search.best_params_.items():
            f.write(f"  • {k:<22}: {v}\n")
        f.write("\n")

        # 4 — Métricas en test set
        f.write(f"{sep}\n")
        f.write("=== MÉTRICAS EN TEST SET (NUNCA VISTO) ===\n")
        f.write(f"{sep}\n")
        f.write(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.1f}%)\n")
        f.write(f"  Precision : {precision:.4f}\n")
        f.write(f"  Recall    : {recall:.4f}\n")
        f.write(f"  F1-Score  : {f1:.4f}\n")
        f.write(f"  ROC-AUC   : {roc_auc:.4f}\n\n")
        f.write("Umbral óptimo (índice de Youden — max TPR-FPR):\n")
        f.write(f"  • Umbral    : {optimal_threshold:.4f}\n")
        f.write(f"  • TPR óptimo: {optimal_tpr:.4f}\n")
        f.write(f"  • FPR óptimo: {optimal_fpr:.4f}\n\n")
        f.write("Cross-Validation (5-fold, train+val conjunto):\n")
        f.write(f"  • F1 promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")

        # 5 — Análisis de errores
        f.write(f"{sep}\n")
        f.write("=== ANÁLISIS DE ERRORES ===\n")
        f.write(f"{sep}\n")
        f.write(f"Total en test set : {total_test} muestras\n\n")
        f.write(
            f"  ✅ Verdaderos Positivos (TP) : {tp_count:>4}  "
            f"({tp_count / total_test:.1%})  — exitoso predicho correctamente\n"
        )
        f.write(
            f"  ✅ Verdaderos Negativos (TN) : {tn_count:>4}  "
            f"({tn_count / total_test:.1%})  — no exitoso predicho correctamente\n"
        )
        f.write(
            f"  ❌ Falsos Positivos   (FP)   : {fp_count:>4}  "
            f"({fp_count / total_test:.1%})  — predijo éxito, era fracaso\n"
        )
        f.write(
            f"  ❌ Falsos Negativos   (FN)   : {fn_count:>4}  "
            f"({fn_count / total_test:.1%})  — predijo fracaso, era éxito\n\n"
        )
        f.write(f"Tasa de error total : {(fp_count + fn_count) / total_test:.1%}\n")
        f.write(
            "  • Los FP generan sobre-optimismo: el equipo cree que alguien es productivo\n"
        )
        f.write(
            "    cuando en realidad no lo es → riesgo de asignación errónea de tareas.\n"
        )
        f.write(
            "  • Los FN representan talento no detectado → oportunidades perdidas de\n"
        )
        f.write("    reconocer y potenciar a desarrolladores productivos.\n\n")

        # 6 — Feature importance top 5
        f.write(f"{sep}\n")
        f.write("=== FEATURE IMPORTANCE — TOP 5 FACTORES ===\n")
        f.write(f"{sep}\n")
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            bar = "█" * int(row["importance"] * 200)
            f.write(
                f"  {rank}. {row['label']:<26} {row['importance']:.4f}  "
                f"({row['importance'] * 100:.1f}%)  {bar}\n"
            )
        f.write("\n")

        # 7 — Interpretación de resultados
        f.write(f"{sep}\n")
        f.write("=== INTERPRETACIÓN DE RESULTADOS ===\n")
        f.write(f"{sep}\n")

        top1_row = feat_imp_df.nlargest(1, "importance").iloc[0]
        f.write(f"1. FACTOR MÁS DETERMINANTE:\n")
        f.write(
            f"   '{top1_row['label']}' explica el {top1_row['importance'] * 100:.1f}% de la varianza\n"
        )
        f.write(
            f"   del modelo. Optimizar esta variable tiene el mayor impacto en task_success.\n\n"
        )

        f.write(f"2. BALANCE PRECISION / RECALL:\n")
        f.write(f"   Precision={precision:.2f}  vs  Recall={recall:.2f}. ")
        if precision > recall + 0.05:
            f.write(
                "El modelo es conservador (pocos FP): cuando predice éxito, suele\n"
            )
            f.write("   acertar, pero puede pasar por alto casos reales de éxito.\n\n")
        elif recall > precision + 0.05:
            f.write("El modelo es agresivo (pocos FN): captura casi todos los casos\n")
            f.write("   de éxito real, pero con mayor tasa de falsas alarmas.\n\n")
        else:
            f.write("Modelo bien balanceado. Distribución equitativa de errores.\n\n")

        f.write(f"3. ROBUSTEZ CV vs TEST:\n")
        diff = abs(cv_scores.mean() - f1)
        f.write(
            f"   CV F1={cv_scores.mean():.3f}  vs  Test F1={f1:.3f}  (Δ={diff:.3f}). "
        )
        if diff < 0.03:
            f.write("Diferencia mínima. El modelo generaliza excelentemente.\n\n")
        elif diff < 0.06:
            f.write("Diferencia leve. El modelo generaliza bien.\n\n")
        else:
            f.write("Diferencia notable. Considerar regularización adicional.\n\n")

        f.write(f"4. RECOMENDACIONES ACCIONABLES (basadas en top features):\n")
        for i, (_, row) in enumerate(top5.head(3).iterrows(), 1):
            f.write(f"   {i}. Monitorear y optimizar '{row['label']}'\n")
        f.write("\n")

        # 8 — Limitaciones
        f.write(f"{sep}\n")
        f.write("=== LIMITACIONES DEL MODELO ===\n")
        f.write(f"{sep}\n")
        f.write("• Dataset limitado (~500 registros): puede no generalizar\n")
        f.write("  a todos los equipos o contextos de desarrollo.\n")
        f.write("• Las features categóricas (coffee_category, sleep_category)\n")
        f.write("  fueron codificadas numéricamente; el modelo no captura\n")
        f.write("  la naturaleza ordinal de forma explícita.\n")
        f.write("• GridSearchCV exploró un subespacio reducido: pueden existir\n")
        f.write("  combinaciones de hiperparámetros mejores no exploradas.\n")
        f.write("• No se aplicó balanceo de clases (SMOTE, class_weight);\n")
        f.write(
            "  si el dataset es desbalanceado, las métricas pueden estar infladas.\n"
        )
        f.write("• La importancia Gini puede sobreestimar features con muchos\n")
        f.write("  valores únicos (continuas de alta cardinalidad).\n")
        f.write("• El modelo predice task_success como binario; no captura\n")
        f.write("  niveles intermedios de productividad.\n")
        f.write(f"\nGenerado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"   ✅ fase45_estadisticas.txt guardado")


# ============================================================
# Almacén temporal para compartir DataFrames entre funciones
# (evita pasar 6 variables por cada función de visualización)
# ============================================================
_GLOBAL: dict = {}


# ============================================================
# main()
# ============================================================


def main():
    _print_separator()
    print("🚀 Fase IV-V: Modelado y Evaluación de Productividad")
    _print_separator()

    # Crear directorios de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_MODEL, exist_ok=True)
    os.makedirs(DASHBOARD_ASSETS, exist_ok=True)

    # --- Paso 1: Cargar datos ---
    X_train, X_val, X_test, y_train, y_val, y_test, prep_info = load_data()

    # Guardar referencias globales para funciones que las necesitan
    _GLOBAL["X_train"] = X_train
    _GLOBAL["X_val"] = X_val
    _GLOBAL["X_test"] = X_test
    _GLOBAL["y_train"] = y_train
    _GLOBAL["y_val"] = y_val
    _GLOBAL["y_test"] = y_test

    # --- Paso 2: Escalar ---
    scaler, X_train_sc, X_val_sc, X_test_sc = scale_features(X_train, X_val, X_test)

    # --- Paso 3: Baseline ---
    models = train_baselines(X_train_sc, y_train)

    # --- Paso 4: Evaluación en val ---
    val_metrics = evaluate_on_val(models, X_val_sc, y_val)

    # --- Paso 5: Optimización RF ---
    grid_search, best_model = optimize_random_forest(X_train_sc, y_train)

    # --- Paso 6: Evaluación en test ---
    y_pred, y_prob, cm, test_metrics, cv_scores = evaluate_on_test(
        best_model, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test
    )

    # --- Paso 7: Visualizaciones ---
    print("\n🎨 Paso 7: Generando visualizaciones...")
    plt.style.use("default")
    sns.set_palette("husl")

    plot_model_comparison(val_metrics)
    plot_confusion_matrix(cm, test_metrics["accuracy"])
    _, optimal_threshold, optimal_fpr, optimal_tpr = plot_roc_curve(
        y_test, y_prob, test_metrics["roc_auc"]
    )
    _, feat_imp_df, top3_features = plot_feature_importance(
        best_model, test_metrics["accuracy"]
    )
    _, error_types = plot_error_analysis(X_test, y_test, y_pred, feat_imp_df)
    print("   ✅ Todas las visualizaciones generadas")

    # --- Paso 8: Pipeline + metadata ---
    save_pipeline_and_metadata(best_model, X_train, X_val, y_train, y_val, test_metrics)

    # --- Paso 9: Copiar imágenes al dashboard ---
    copy_to_dashboard()

    # --- Paso 10: Estadísticas ---
    write_statistics(
        val_metrics,
        grid_search,
        best_model,
        test_metrics,
        cv_scores,
        cm,
        feat_imp_df,
        error_types,
        optimal_threshold,
        optimal_fpr,
        optimal_tpr,
    )

    # --- Paso 11: Resumen final en consola ---
    print()
    _print_separator()
    print("✅ Fase IV-V completada")
    print(f"   Mejor modelo:    Random Forest")
    print(f"   Test Accuracy:   {test_metrics['accuracy'] * 100:.1f}%")
    print(f"   Test F1-Score:   {test_metrics['f1']:.3f}")
    print(f"   Test ROC-AUC:    {test_metrics['roc_auc']:.3f}")
    print(f"   Top 3 features:  {top3_features}")
    print(f"   Modelo guardado: {DASHBOARD_MODEL}/pipeline.pkl")
    _print_separator()


if __name__ == "__main__":
    main()
