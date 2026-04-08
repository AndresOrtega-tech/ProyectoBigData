#!/usr/bin/env python3
"""
Fase III: Preparación de Datos
Pipeline de limpieza, feature engineering, balance de clases y división train/val/test
para el proyecto de productividad de desarrolladores con IA.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Variables originales
    "hours_coding",
    "coffee_intake_mg",
    "distractions",
    "sleep_hours",
    "commits",
    "bugs_reported",
    "ai_usage_hours",
    "cognitive_load",
    # Variables engineered
    "sleep_deficit",
    "productivity_ratio",
    "caffeine_per_hour",
    "work_intensity",
    "coffee_category",
    "sleep_category",
]
TARGET = "task_success"

ORIGINAL_FEATURES = [
    "hours_coding",
    "coffee_intake_mg",
    "distractions",
    "sleep_hours",
    "commits",
    "bugs_reported",
    "ai_usage_hours",
    "cognitive_load",
]

OUTPUT_DIR = "results/fase3-preparacion"


# ──────────────────────────────────────────────────────────────────────────────
# Paso 1 — Cargar datos
# ──────────────────────────────────────────────────────────────────────────────


def cargar_datos() -> pd.DataFrame:
    """Carga el dataset desde ruta relativa con fallback a ruta absoluta."""
    rutas = [
        "../data/ai_dev_productivity.csv",
        "/Users/andrestamez5/Personal/BigDataProject/data/ai_dev_productivity.csv",
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            df = pd.read_csv(ruta)
            print(f"   📁 Dataset cargado desde: {ruta}")
            print(f"   Registros: {len(df)} | Columnas: {len(df.columns)}")
            return df

    raise FileNotFoundError(
        "No se encuentra el dataset en ninguna ruta conocida. "
        f"Rutas intentadas: {rutas}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — Limpieza de datos
# ──────────────────────────────────────────────────────────────────────────────


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el pipeline de limpieza:
      - Imputación/eliminación de nulos
      - Detección y eliminación de outliers críticos
      - Validación de consistencia lógica
    Retorna el dataframe limpio y un resumen del proceso.
    """
    reporte = {
        "nulos_por_columna": {},
        "filas_eliminadas_nulos": 0,
        "filas_eliminadas_outliers": 0,
        "filas_eliminadas_inconsistencias": 0,
        "outliers_detectados_iqr": {},
        "columnas_imputadas": [],
    }

    n_inicio = len(df)

    # ── 2.1 Valores nulos ────────────────────────────────────────────────────
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df)) * 100

    for col_name in df.columns:
        n_nulos = int(nulos[col_name])
        pct = float(nulos_pct[col_name])
        reporte["nulos_por_columna"][col_name] = {
            "count": n_nulos,
            "pct": round(pct, 2),
        }
        if n_nulos == 0:
            continue
        if pct < 5.0:
            # Imputar con mediana para columnas numéricas
            if df[col_name].dtype in [np.float64, np.int64, float, int]:
                mediana = df[col_name].median()
                df[col_name] = df[col_name].fillna(mediana)
                reporte["columnas_imputadas"].append(
                    {
                        "columna": col_name,
                        "mediana": float(mediana),
                        "n_imputados": n_nulos,
                    }
                )
                print(
                    f"   ⚠️  Nulos en '{col_name}': {n_nulos} ({pct:.1f}%) → imputados con mediana ({mediana:.2f})"
                )
        else:
            # ≥5%: reportar y eliminar filas
            print(
                f"   ❌ Nulos en '{col_name}': {n_nulos} ({pct:.1f}%) ≥ 5% → eliminando filas"
            )
            df = df.dropna(subset=[col_name])

    filas_eliminadas_nulos = n_inicio - len(df)
    reporte["filas_eliminadas_nulos"] = filas_eliminadas_nulos
    if filas_eliminadas_nulos > 0:
        print(f"   🗑️  Filas eliminadas por nulos: {filas_eliminadas_nulos}")

    # ── 2.2 Detección de outliers por IQR ───────────────────────────────────
    n_antes_outliers = len(df)
    mask_eliminar = pd.Series([False] * len(df), index=df.index)

    for col_name in ORIGINAL_FEATURES:
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((df[col_name] < lower) | (df[col_name] > upper)).sum())
        reporte["outliers_detectados_iqr"][col_name] = {
            "n_outliers": n_outliers,
            "lower_fence": round(float(lower), 4),
            "upper_fence": round(float(upper), 4),
        }

    # Outliers críticos → eliminación directa (inconsistencias lógicas severas)
    mask_hours = df["hours_coding"] > 20
    mask_coffee = df["coffee_intake_mg"] > 1500
    mask_sleep = (df["sleep_hours"] > 16) | (df["sleep_hours"] < 0)

    n_hours = int(mask_hours.sum())
    n_coffee = int(mask_coffee.sum())
    n_sleep = int(mask_sleep.sum())

    if n_hours > 0:
        print(
            f"   🗑️  hours_coding > 20: {n_hours} fila(s) → eliminadas (inconsistencia lógica)"
        )
    if n_coffee > 0:
        print(
            f"   🗑️  coffee_intake_mg > 1500: {n_coffee} fila(s) → eliminadas (posible error)"
        )
    if n_sleep > 0:
        print(
            f"   🗑️  sleep_hours fuera de [0, 16]: {n_sleep} fila(s) → eliminadas (imposible)"
        )

    mask_eliminar = mask_hours | mask_coffee | mask_sleep
    df = df[~mask_eliminar].reset_index(drop=True)

    reporte["filas_eliminadas_outliers"] = n_antes_outliers - len(df)

    # ── 2.3 Inconsistencias lógicas adicionales ──────────────────────────────
    n_antes_inc = len(df)
    mask_inc = (
        (df["hours_coding"] < 0)
        | (df["coffee_intake_mg"] < 0)
        | (~df["task_success"].isin([0, 1]))
    )
    n_inconsistencias = int(mask_inc.sum())
    if n_inconsistencias > 0:
        print(
            f"   🗑️  Inconsistencias lógicas adicionales: {n_inconsistencias} fila(s) → eliminadas"
        )
        df = df[~mask_inc].reset_index(drop=True)

    reporte["filas_eliminadas_inconsistencias"] = n_antes_inc - len(df)

    total_eliminadas = n_inicio - len(df)
    print(f"\n   📊 Resumen limpieza:")
    print(f"      Filas originales:     {n_inicio}")
    print(f"      Filas eliminadas:     {total_eliminadas}")
    print(f"      Filas resultantes:    {len(df)}")

    return df, reporte


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3 — Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────


def hacer_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea 6 variables derivadas con justificación teórica:
      - sleep_deficit: cuánto le falta al desarrollador respecto a las 8h recomendadas
      - productivity_ratio: eficiencia por hora de código (commits / hora)
      - caffeine_per_hour: intensidad de consumo de cafeína relativa al tiempo codificando
      - work_intensity: proxy de fatiga cognitiva acumulada (horas × carga)
      - coffee_category: segmentación ordinal de consumo (bajo/medio/alto)
      - sleep_category: segmentación ordinal de descanso (insuficiente/óptimo/excesivo)
    """
    # Déficit de sueño respecto a las 8h recomendadas
    df["sleep_deficit"] = 8.0 - df["sleep_hours"]

    # Ratio de productividad: commits por hora de código (evitar división por cero)
    df["productivity_ratio"] = df["commits"] / df["hours_coding"].replace(0, 0.1)

    # Cafeína por hora de código
    df["caffeine_per_hour"] = df["coffee_intake_mg"] / df["hours_coding"].replace(
        0, 0.1
    )

    # Intensidad de trabajo: horas × carga cognitiva
    df["work_intensity"] = df["hours_coding"] * df["cognitive_load"]

    # Categoría de cafeína (numérica: 0=bajo, 1=medio, 2=alto)
    df["coffee_category"] = pd.cut(
        df["coffee_intake_mg"],
        bins=[-1, 200, 400, 2000],
        labels=[0, 1, 2],
    ).astype(int)

    # Categoría de sueño (numérica: 0=insuficiente, 1=óptimo, 2=excesivo)
    df["sleep_category"] = pd.cut(
        df["sleep_hours"],
        bins=[-1, 6, 8, 25],
        labels=[0, 1, 2],
    ).astype(int)

    print(f"   ✅ 6 features engineered creadas")
    print(f"      sleep_deficit     | Diferencia respecto a 8h de sueño recomendadas")
    print(f"      productivity_ratio| Commits / horas_codigo (eficiencia)")
    print(f"      caffeine_per_hour | mg cafeína / horas_codigo (intensidad consumo)")
    print(
        f"      work_intensity    | horas_codigo × carga_cognitiva (fatiga acumulada)"
    )
    print(
        f"      coffee_category   | Segmentación ordinal cafeína (0=bajo, 1=medio, 2=alto)"
    )
    print(
        f"      sleep_category    | Segmentación ordinal sueño (0=insuf., 1=óptimo, 2=excesivo)"
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Paso 5 — Balance de clases
# ──────────────────────────────────────────────────────────────────────────────


def verificar_balance(y: pd.Series) -> tuple[float, dict]:
    """
    Calcula el porcentaje de la clase minoritaria y la distribución completa.
    """
    conteo = y.value_counts()
    total = len(y)
    pct_minoritaria = float((conteo.min() / total) * 100)
    distribucion = {str(k): int(v) for k, v in conteo.items()}
    return pct_minoritaria, distribucion


# ──────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ──────────────────────────────────────────────────────────────────────────────


def graficar_distribucion_target(
    y_original: pd.Series,
    y_train_final: pd.Series,
    smote_aplicado: bool,
    output_dir: str,
) -> None:
    """
    Genera pie chart + barplot de distribución de task_success.
    Si se aplicó SMOTE, compara distribución antes y después en training.
    """
    fig, axes = plt.subplots(
        1, 3 if smote_aplicado else 2, figsize=(16 if smote_aplicado else 12, 5)
    )
    fig.suptitle("Distribución de task_success", fontsize=14, fontweight="bold", y=1.02)

    colores = ["#E74C3C", "#2ECC71"]
    etiquetas = ["No éxito (0)", "Éxito (1)"]

    # Pie chart distribución original
    conteo_orig = y_original.value_counts().sort_index()
    axes[0].pie(
        conteo_orig.values,
        labels=etiquetas,
        colors=colores,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[0].set_title("Distribución original\n(dataset completo)", fontsize=11)

    # Barplot distribución original
    conteo_df = pd.DataFrame(
        {
            "Clase": [f"Clase {k}" for k in conteo_orig.index],
            "Conteo": conteo_orig.values,
        }
    )
    barplot = sns.barplot(
        data=conteo_df,
        x="Clase",
        y="Conteo",
        palette=colores,
        ax=axes[1],
        hue="Clase",
        legend=False,
    )
    for p in barplot.patches:
        barplot.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    axes[1].set_title("Conteo original\n(dataset completo)", fontsize=11)
    axes[1].set_xlabel("Clase")
    axes[1].set_ylabel("Cantidad de registros")

    # Si se aplicó SMOTE: barplot post-SMOTE en training
    if smote_aplicado:
        conteo_smote = y_train_final.value_counts().sort_index()
        conteo_smote_df = pd.DataFrame(
            {
                "Clase": [f"Clase {k}" for k in conteo_smote.index],
                "Conteo": conteo_smote.values,
            }
        )
        barplot2 = sns.barplot(
            data=conteo_smote_df,
            x="Clase",
            y="Conteo",
            palette=["#3498DB", "#F39C12"],
            ax=axes[2],
            hue="Clase",
            legend=False,
        )
        for p in barplot2.patches:
            barplot2.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        axes[2].set_title("Conteo post-SMOTE\n(solo training)", fontsize=11)
        axes[2].set_xlabel("Clase")
        axes[2].set_ylabel("Cantidad de registros")

    plt.tight_layout()
    ruta = os.path.join(output_dir, "fase3_distribucion_target.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Guardado: {ruta}")


def graficar_outliers(df: pd.DataFrame, output_dir: str) -> None:
    """
    Genera boxplots de las 8 variables originales en grid 2×4.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "Distribución y Outliers — Variables Originales", fontsize=14, fontweight="bold"
    )

    for idx, col_name in enumerate(ORIGINAL_FEATURES):
        ax = axes[idx // 4][idx % 4]
        sns.boxplot(
            y=df[col_name],
            ax=ax,
            color="#5DADE2",
            flierprops={
                "marker": "o",
                "markerfacecolor": "#E74C3C",
                "markersize": 4,
                "alpha": 0.6,
            },
            width=0.5,
        )
        ax.set_title(col_name, fontsize=10, fontweight="bold")
        ax.set_ylabel("")

        # Anotar estadísticas clave
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        n_out = int(
            ((df[col_name] < q1 - 1.5 * iqr) | (df[col_name] > q3 + 1.5 * iqr)).sum()
        )
        ax.set_xlabel(f"IQR={iqr:.1f} | Outliers={n_out}", fontsize=8, color="#7F8C8D")

    plt.tight_layout()
    ruta = os.path.join(output_dir, "fase3_outliers.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Guardado: {ruta}")


def graficar_correlation_matrix(df: pd.DataFrame, output_dir: str) -> None:
    """
    Genera heatmap de correlación de las 14 features + target con anotaciones.
    """
    cols_para_correlacion = FEATURE_NAMES + [TARGET]
    corr_df = df[cols_para_correlacion].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.zeros_like(corr_df, dtype=bool)
    # Sin máscara: mostrar matriz completa para ver correlación con target

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_df,
        ax=ax,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.5,
        linecolor="#BDC3C7",
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    ax.set_title(
        "Matriz de Correlación — Features + Target\n(14 variables + task_success)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    ruta = os.path.join(output_dir, "fase3_correlation_matrix.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────────
# Paso 7 — Guardar resultados
# ──────────────────────────────────────────────────────────────────────────────


def guardar_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """Guarda los 6 splits en CSV para que fase45 los cargue directamente."""
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"   💾 6 splits CSV guardados en {output_dir}/")


def guardar_metadata(
    info: dict,
    output_dir: str,
) -> None:
    """Guarda preprocessing_info.json con toda la metadata del pipeline."""
    ruta = os.path.join(output_dir, "preprocessing_info.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"   💾 Guardado: {ruta}")


def guardar_estadisticas(
    df_original: pd.DataFrame,
    df_limpio: pd.DataFrame,
    reporte_limpieza: dict,
    dist_antes: dict,
    dist_despues_train: dict,
    smote_aplicado: bool,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str,
) -> None:
    """Genera el archivo de estadísticas con las 8 secciones requeridas."""
    ruta = os.path.join(output_dir, "fase3_estadisticas.txt")

    with open(ruta, "w", encoding="utf-8") as f:
        # ── Sección 1 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== FASE III: PREPARACIÓN DE DATOS ===\n")
        f.write("=" * 60 + "\n")
        f.write("Proyecto: Productividad de Desarrolladores con IA\n")
        f.write(f"Dataset:  ai_dev_productivity.csv\n")
        f.write(f"Registros originales: {len(df_original)}\n")
        f.write(f"Registros post-limpieza: {len(df_limpio)}\n\n")

        # ── Sección 2 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== ANÁLISIS DE CALIDAD DE DATOS ===\n")
        f.write("=" * 60 + "\n\n")

        f.write(">> Valores nulos por columna:\n")
        for col_name, info_nulos in reporte_limpieza["nulos_por_columna"].items():
            estado = ""
            if info_nulos["count"] == 0:
                estado = "OK"
            elif info_nulos["pct"] < 5:
                estado = f"imputado con mediana"
            else:
                estado = "filas eliminadas"
            f.write(
                f"   {col_name:<22}: {info_nulos['count']:>4} nulos ({info_nulos['pct']:>5.1f}%)  → {estado}\n"
            )

        f.write(
            f"\n>> Filas eliminadas por nulos:           {reporte_limpieza['filas_eliminadas_nulos']}\n"
        )
        f.write(
            f">> Filas eliminadas por outliers críticos:{reporte_limpieza['filas_eliminadas_outliers']}\n"
        )
        f.write(
            f">> Filas eliminadas por inconsistencias:  {reporte_limpieza['filas_eliminadas_inconsistencias']}\n\n"
        )

        f.write(">> Outliers detectados por IQR (1.5×IQR):\n")
        for col_name, info_iqr in reporte_limpieza["outliers_detectados_iqr"].items():
            f.write(
                f"   {col_name:<22}: {info_iqr['n_outliers']:>3} outliers "
                f"[fence: {info_iqr['lower_fence']:.2f}, {info_iqr['upper_fence']:.2f}]\n"
            )
        f.write(
            "\n   Nota: outliers IQR que NO son inconsistencias lógicas se mantienen\n"
        )
        f.write("         (pueden representar casos extremos legítimos).\n\n")

        # ── Sección 3 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== FEATURE ENGINEERING ===\n")
        f.write("=" * 60 + "\n\n")
        features_info = [
            (
                "sleep_deficit",
                "8.0 - sleep_hours",
                "Cuantifica privación de sueño respecto a las 8h recomendadas",
            ),
            (
                "productivity_ratio",
                "commits / hours_coding",
                "Eficiencia por hora de código; evita sesgo por duración de sesión",
            ),
            (
                "caffeine_per_hour",
                "coffee_intake_mg / hours_coding",
                "Intensidad de consumo relativa al tiempo activo",
            ),
            (
                "work_intensity",
                "hours_coding × cognitive_load",
                "Proxy de fatiga cognitiva acumulada en la sesión",
            ),
            (
                "coffee_category",
                "pd.cut([bajo, medio, alto])",
                "Segmentación ordinal: 0=<200mg, 1=200-400mg, 2=>400mg",
            ),
            (
                "sleep_category",
                "pd.cut([insuf., óptimo, excesivo])",
                "Segmentación ordinal: 0=<6h, 1=6-8h, 2=>8h",
            ),
        ]
        for nombre, formula, justificacion in features_info:
            f.write(f"   {nombre:<22} = {formula}\n")
            f.write(f"   {'':22}  → {justificacion}\n\n")

        # ── Sección 4 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS PRE-PROCESAMIENTO ===\n")
        f.write("=" * 60 + "\n\n")
        stats_pre = df_original[ORIGINAL_FEATURES + [TARGET]].describe().round(4)
        f.write(stats_pre.to_string())
        f.write("\n\n")

        # ── Sección 5 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS POST-PROCESAMIENTO ===\n")
        f.write("=" * 60 + "\n\n")
        stats_post = df_limpio[FEATURE_NAMES + [TARGET]].describe().round(4)
        f.write(stats_post.to_string())
        f.write("\n\n")

        # ── Sección 6 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== BALANCE DE CLASES ===\n")
        f.write("=" * 60 + "\n\n")
        total_orig = sum(dist_antes.values())
        f.write(">> Distribución original (dataset completo post-limpieza):\n")
        for clase, conteo in sorted(dist_antes.items()):
            pct = (conteo / total_orig) * 100
            f.write(f"   Clase {clase}: {conteo:>5} registros ({pct:.1f}%)\n")

        f.write(f"\n>> SMOTE aplicado: {'Sí' if smote_aplicado else 'No'}\n")
        if smote_aplicado:
            total_post = sum(dist_despues_train.values())
            f.write(">> Distribución post-SMOTE (solo training):\n")
            for clase, conteo in sorted(dist_despues_train.items()):
                pct = (conteo / total_post) * 100
                f.write(f"   Clase {clase}: {conteo:>5} registros ({pct:.1f}%)\n")
        f.write("\n")

        # ── Sección 7 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== DIVISIÓN TRAIN/VAL/TEST ===\n")
        f.write("=" * 60 + "\n\n")
        total_splits = len(X_train) + len(X_val) + len(X_test)
        f.write(f"   Estrategia:  70% train / 15% val / 15% test (estratificado)\n")
        f.write(f"   random_state: 42\n\n")
        f.write(
            f"   Train: {len(X_train):>5} registros ({len(X_train) / total_splits * 100:.1f}%)\n"
        )
        f.write(
            f"   Val:   {len(X_val):>5} registros ({len(X_val) / total_splits * 100:.1f}%)\n"
        )
        f.write(
            f"   Test:  {len(X_test):>5} registros ({len(X_test) / total_splits * 100:.1f}%)\n"
        )
        f.write(
            f"\n   Features usadas ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}\n"
        )
        f.write(f"   Target: {TARGET}\n\n")

        # ── Sección 8 ──────────────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write("=== RESUMEN EJECUTIVO ===\n")
        f.write("=" * 60 + "\n\n")
        total_eliminadas = len(df_original) - len(df_limpio)
        f.write(f"   Registros originales:         {len(df_original)}\n")
        f.write(
            f"   Registros eliminados:         {total_eliminadas} ({total_eliminadas / len(df_original) * 100:.1f}%)\n"
        )
        f.write(f"   Registros limpios:            {len(df_limpio)}\n")
        f.write(f"   Variables originales:         {len(ORIGINAL_FEATURES)}\n")
        f.write(f"   Variables engineered:         6 nuevas\n")
        f.write(f"   Total features para modelo:   {len(FEATURE_NAMES)}\n")
        f.write(
            f"   SMOTE aplicado:               {'Sí' if smote_aplicado else 'No'}\n"
        )
        f.write(f"   Train size (final):           {len(X_train)}\n")
        f.write(f"   Val size:                     {len(X_val)}\n")
        f.write(f"   Test size:                    {len(X_test)}\n")
        f.write(f"\n   Archivos generados:\n")
        archivos = [
            "X_train.csv",
            "X_val.csv",
            "X_test.csv",
            "y_train.csv",
            "y_val.csv",
            "y_test.csv",
            "preprocessing_info.json",
            "fase3_estadisticas.txt",
            "fase3_distribucion_target.png",
            "fase3_outliers.png",
            "fase3_correlation_matrix.png",
        ]
        for archivo in archivos:
            f.write(f"      - notebooks/results/fase3-preparacion/{archivo}\n")

    print(f"   💾 Guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────────
# main()
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("🔧 Iniciando Fase III: Preparación de Datos")
    print("=" * 60)

    # Crear directorio de salida (idempotente)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Paso 1: Cargar datos ─────────────────────────────────────────────────
    print("\n📂 Paso 1: Cargando datos...")
    df = cargar_datos()
    n_original = len(df)
    df_original = df.copy()  # guardar copia pre-limpieza para estadísticas

    # ── Paso 2: Limpieza ─────────────────────────────────────────────────────
    print("\n🧹 Paso 2: Limpieza de datos...")
    df, reporte_limpieza = limpiar_datos(df)
    n_clean = len(df)

    # ── Paso 3: Feature Engineering ─────────────────────────────────────────
    print("\n⚙️  Paso 3: Feature Engineering...")
    df = hacer_feature_engineering(df)
    df_limpio_con_features = df.copy()  # guardar para estadísticas post-procesamiento

    # ── Paso 4: Definir features ─────────────────────────────────────────────
    print("\n🎯 Paso 4: Verificando features...")
    features_faltantes = [f for f in FEATURE_NAMES if f not in df.columns]
    if features_faltantes:
        raise ValueError(f"Features faltantes en el DataFrame: {features_faltantes}")
    print(f"   ✅ {len(FEATURE_NAMES)} features verificadas | Target: {TARGET}")

    X = df[FEATURE_NAMES].copy()
    y = df[TARGET].copy()

    # ── Paso 5: Balance de clases (evaluación pre-split) ─────────────────────
    print("\n⚖️  Paso 5: Evaluando balance de clases...")
    clase_minoritaria_pct, dist_original = verificar_balance(y)
    print(f"   Distribución task_success:")
    for clase, conteo in sorted(dist_original.items()):
        pct = (conteo / len(y)) * 100
        print(f"      Clase {clase}: {conteo} registros ({pct:.1f}%)")
    print(f"   Clase minoritaria: {clase_minoritaria_pct:.1f}% del total")

    aplicar_smote = clase_minoritaria_pct < 40.0
    if aplicar_smote:
        print(f"   ⚠️  Clase minoritaria < 40% → se aplicará SMOTE al training set")
    else:
        print(f"   ✅ Clases suficientemente balanceadas → SMOTE no requerido")

    # ── Paso 6: División Train/Val/Test ──────────────────────────────────────
    print("\n✂️  Paso 6: División Train/Val/Test (70/15/15 estratificado)...")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=y_train_val,
    )

    print(
        f"   Pre-SMOTE  → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # SMOTE solo en training
    smote_aplicado = False
    dist_train_pre_smote = {str(k): int(v) for k, v in y_train.value_counts().items()}
    dist_train_post_smote = dist_train_pre_smote.copy()

    if aplicar_smote:
        print("   🔄 Aplicando SMOTE al training set...")
        smote = SMOTE(random_state=42)
        X_train_np, y_train_np = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_np, columns=FEATURE_NAMES)
        y_train = pd.Series(y_train_np, name=TARGET)
        smote_aplicado = True
        dist_train_post_smote = {
            str(k): int(v) for k, v in y_train.value_counts().items()
        }
        print(
            f"   Post-SMOTE → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
        )
        print(f"   Distribución post-SMOTE (training):")
        for clase, conteo in sorted(dist_train_post_smote.items()):
            pct = (conteo / len(y_train)) * 100
            print(f"      Clase {clase}: {conteo} ({pct:.1f}%)")

    # ── Paso 7: Guardar resultados ────────────────────────────────────────────
    print(f"\n💾 Paso 7: Guardando resultados en {OUTPUT_DIR}/...")

    # 7.1 Splits CSV
    guardar_splits(X_train, X_val, X_test, y_train, y_val, y_test, OUTPUT_DIR)

    # 7.2 Metadata JSON
    info = {
        "feature_names": FEATURE_NAMES,
        "total_original": int(n_original),
        "total_after_cleaning": int(n_clean),
        "rows_removed": int(n_original - n_clean),
        "smote_applied": bool(smote_aplicado),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "class_distribution_train": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        },
        "class_distribution_test": {
            "0": int((y_test == 0).sum()),
            "1": int((y_test == 1).sum()),
        },
    }
    guardar_metadata(info, OUTPUT_DIR)

    # 7.3 Estadísticas texto
    guardar_estadisticas(
        df_original=df_original,
        df_limpio=df_limpio_con_features,
        reporte_limpieza=reporte_limpieza,
        dist_antes=dist_original,
        dist_despues_train=dist_train_post_smote,
        smote_aplicado=smote_aplicado,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        output_dir=OUTPUT_DIR,
    )

    # 7.4 Visualizaciones PNG
    print("\n📊 Generando visualizaciones...")
    graficar_distribucion_target(
        y_original=y,
        y_train_final=y_train,
        smote_aplicado=smote_aplicado,
        output_dir=OUTPUT_DIR,
    )
    graficar_outliers(df=df_original, output_dir=OUTPUT_DIR)
    graficar_correlation_matrix(df=df_limpio_con_features, output_dir=OUTPUT_DIR)

    # ── Paso 8: Output final ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Fase III completada")
    print(f"   Registros originales:  {n_original}")
    print(f"   Registros limpios:     {n_clean}")
    print(f"   Features creadas:      6 nuevas (14 total)")
    print(f"   SMOTE aplicado:        {'Sí' if smote_aplicado else 'No'}")
    print(
        f"   Train:  {len(X_train)} registros | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    print(f"   Archivos guardados en {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
