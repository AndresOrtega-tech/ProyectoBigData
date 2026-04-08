#!/usr/bin/env python3
"""
Plan 8: Análisis Multivariado — Balance Óptimo Cafeína + Horas + Sueño

Hipótesis: "Existe un balance óptimo entre cafeína (coffee_intake_mg),
horas de código (hours_coding) y sueño (sleep_hours) que maximiza la tasa
de task_success, superior al efecto individual de cada variable."
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, when

# ─── Constantes ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "notebooks/results/plan8-balance-optimo"

# Correlaciones individuales ya conocidas del proyecto
CORR_CAFEINA = 0.695
CORR_HORAS = 0.616
CORR_SUENO = 0.187

# Orden canónico para heatmaps
ORDEN_RANGOS = ["bajo", "medio", "alto"]

# Alias legibles para ejes con equivalencias prácticas
ALIAS_CAFEINA = {
    "bajo": "BAJO\n(<200 mg)\n~<2 tazas",
    "medio": "MEDIO\n(200-400 mg)\n~2-4 tazas",
    "alto": "ALTO\n(>400 mg)\n~>4 tazas",
}
ALIAS_HORAS = {
    "bajo": "BAJO\n(<4 h/día)",
    "medio": "MEDIO\n(4-7 h/día)",
    "alto": "ALTO\n(>7 h/día)",
}
ALIAS_SUENO = {
    "bajo": "BAJO\n(<6 h/noche)",
    "medio": "MEDIO\n(6-8 h/noche)\nRecomendado",
    "alto": "ALTO\n(>8 h/noche)",
}


# ─── Funciones auxiliares ─────────────────────────────────────────────────────


def normalizar(serie: pd.Series) -> pd.Series:
    """Normalización min-max → escala al rango [0, 1]."""
    return (serie - serie.min()) / (serie.max() - serie.min())


def agregar_rangos_spark(df_spark):
    """
    Agrega tres columnas de rango categórico al DataFrame de Spark:
    cafeina_rango, horas_rango, sueno_rango.
    """
    return (
        df_spark.withColumn(
            "cafeina_rango",
            when(col("coffee_intake_mg") < 200, "bajo")
            .when(col("coffee_intake_mg") <= 400, "medio")
            .otherwise("alto"),
        )
        .withColumn(
            "horas_rango",
            when(col("hours_coding") < 4, "bajo")
            .when(col("hours_coding") <= 7, "medio")
            .otherwise("alto"),
        )
        .withColumn(
            "sueno_rango",
            when(col("sleep_hours") < 6, "bajo")
            .when(col("sleep_hours") <= 8, "medio")
            .otherwise("alto"),
        )
    )


def calcular_score_compuesto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las 3 variables y calcula el score ponderado por las
    correlaciones individuales conocidas.
    Score = 0.695·cafeína_norm + 0.616·horas_norm + 0.187·sueño_norm
    """
    df = df.copy()
    df["cafeina_norm"] = normalizar(df["coffee_intake_mg"])
    df["horas_norm"] = normalizar(df["hours_coding"])
    df["sueno_norm"] = normalizar(df["sleep_hours"])
    df["score_compuesto"] = (
        CORR_CAFEINA * df["cafeina_norm"]
        + CORR_HORAS * df["horas_norm"]
        + CORR_SUENO * df["sueno_norm"]
    )
    return df


def pivot_tasa(df: pd.DataFrame, index_col: str, cols_col: str) -> pd.DataFrame:
    """Pivot de tasa de éxito media para un par de columnas de rango."""
    p = df.pivot_table(
        values="task_success",
        index=index_col,
        columns=cols_col,
        aggfunc="mean",
    )
    return p.reindex(index=ORDEN_RANGOS, columns=ORDEN_RANGOS)


def pivot_conteos(df: pd.DataFrame, index_col: str, cols_col: str) -> pd.DataFrame:
    """Pivot de conteo de registros para un par de columnas de rango."""
    p = df.pivot_table(
        values="task_success",
        index=index_col,
        columns=cols_col,
        aggfunc="count",
    )
    return p.reindex(index=ORDEN_RANGOS, columns=ORDEN_RANGOS)


# ─── Funciones de visualización ───────────────────────────────────────────────


def guardar_heatmap_pares(
    p_tasa: pd.DataFrame,
    p_cnt: pd.DataFrame,
    label_index: str,
    label_cols: str,
    filename: str,
    alias_y: dict,
    alias_x: dict,
) -> None:
    """
    Genera un heatmap RdYlGn de tasa de éxito para un par de rangos.
    Anotaciones: "87.5%\n(n=12)" por celda.
    Marca con borde negro la celda de mayor éxito.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # Construir matriz de anotaciones como porcentaje + conteo
    annot = np.empty(p_tasa.shape, dtype=object)
    for i in range(p_tasa.shape[0]):
        for j in range(p_tasa.shape[1]):
            tasa = p_tasa.iloc[i, j]
            n = p_cnt.iloc[i, j]
            if np.isnan(tasa):
                annot[i, j] = "N/A"
            else:
                cnt_str = f"(n={int(n)})" if not np.isnan(n) else ""
                annot[i, j] = f"{tasa * 100:.1f}%\n{cnt_str}"

    sns.heatmap(
        p_tasa,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.6,
        linecolor="gray",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "Tasa de Éxito (0 = 0 %, 1 = 100 %)"},
    )

    # Identificar celda óptima (máxima tasa)
    max_idx = np.unravel_index(np.nanargmax(p_tasa.values), p_tasa.shape)
    zona_y = ORDEN_RANGOS[max_idx[0]]
    zona_x = ORDEN_RANGOS[max_idx[1]]
    max_tasa = p_tasa.iloc[max_idx[0], max_idx[1]]

    # Borde negro sobre la celda óptima
    ax.add_patch(
        mpatches.Rectangle(
            (max_idx[1], max_idx[0]),
            1,
            1,
            fill=False,
            edgecolor="black",
            lw=3,
            zorder=5,
        )
    )

    # Título descriptivo con zona óptima y referencia a la hipótesis
    ax.set_title(
        f"Tasa de Éxito: {label_index} × {label_cols}\n"
        f"Hipótesis: balance óptimo multivariado > efecto individual\n"
        f"Zona de mayor éxito ★  {label_index} {zona_y.upper()} "
        f"+ {label_cols} {zona_x.upper()} → {max_tasa * 100:.1f}%",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )

    # Etiquetas de ejes con equivalencias prácticas
    ax.set_yticklabels(
        [alias_y.get(r, r) for r in ORDEN_RANGOS],
        rotation=0,
        fontsize=10,
        va="center",
    )
    ax.set_xticklabels(
        [alias_x.get(r, r) for r in ORDEN_RANGOS],
        rotation=0,
        fontsize=10,
        ha="center",
    )
    ax.set_ylabel(label_index, fontsize=12, labelpad=12)
    ax.set_xlabel(label_cols, fontsize=12, labelpad=12)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Guardado: {ruta}")


def guardar_scatter_zona_dorada(df: pd.DataFrame, zona_optima: dict) -> None:
    """
    Scatter hours_coding vs coffee_intake_mg coloreado por task_success.
    Líneas de referencia en los cortes de rangos.
    Recuadro dorado marcando la zona óptima.
    """
    fig, ax = plt.subplots(figsize=(13, 9))

    paleta = {0: "#e74c3c", 1: "#2ecc71"}
    etiquetas = {0: "Fracaso (task_success=0)", 1: "Éxito (task_success=1)"}

    # Scatter por resultado
    for resultado, grupo in df.groupby("task_success"):
        ax.scatter(
            grupo["coffee_intake_mg"],
            grupo["hours_coding"],
            c=paleta[resultado],
            label=etiquetas[resultado],
            alpha=0.55,
            s=55,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    # Límites reales del dataset para posicionar texto
    caf_max = df["coffee_intake_mg"].max()
    hora_max = df["hours_coding"].max()
    hora_min = df["hours_coding"].min()

    # Líneas de referencia para rangos de cafeína (eje X)
    for umbral, etiq in [(200, "200 mg\nbajo→medio"), (400, "400 mg\nmedio→alto")]:
        ax.axvline(x=umbral, color="#e67e22", linestyle="--", alpha=0.75, linewidth=1.6)
        ax.text(
            umbral,
            hora_max + (hora_max - hora_min) * 0.02,
            etiq,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#e67e22",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Líneas de referencia para rangos de horas (eje Y)
    for umbral, etiq in [(4, "4 h  bajo→medio"), (7, "7 h  medio→alto")]:
        ax.axhline(y=umbral, color="#3498db", linestyle="--", alpha=0.75, linewidth=1.6)
        ax.text(
            caf_max + 5,
            umbral,
            etiq,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#3498db",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Coordenadas del rectángulo zona dorada
    lim_cafeina = {"bajo": (0, 200), "medio": (200, 400), "alto": (400, caf_max + 20)}
    lim_horas = {
        "bajo": (hora_min - 0.3, 4),
        "medio": (4, 7),
        "alto": (7, hora_max + 0.3),
    }

    c_rng = zona_optima["cafeina_rango"]
    h_rng = zona_optima["horas_rango"]
    x0, x1 = lim_cafeina.get(c_rng, (0, caf_max))
    y0, y1 = lim_horas.get(h_rng, (hora_min, hora_max))

    rect = mpatches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=3,
        edgecolor="gold",
        facecolor="yellow",
        alpha=0.18,
        zorder=2,
    )
    ax.add_patch(rect)

    # Anotación zona dorada
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    offset_x = min(cx + 90, caf_max - 60)
    offset_y = min(cy + 1.5, hora_max - 0.5)
    ax.annotate(
        f"★ ZONA DORADA\n"
        f"Cafeína: {c_rng.upper()}\n"
        f"Horas:   {h_rng.upper()}\n"
        f"Sueño:   {zona_optima['sueno_rango'].upper()}\n"
        f"Éxito:   {zona_optima['tasa'] * 100:.1f}%  (n={zona_optima['n']})",
        xy=(cx, cy),
        xytext=(offset_x, offset_y),
        fontsize=9.5,
        fontweight="bold",
        color="#2c3e50",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow",
            edgecolor="gold",
            linewidth=2,
        ),
        arrowprops=dict(arrowstyle="->", color="gold", lw=2),
        zorder=6,
    )

    ax.set_title(
        "Scatter: Horas de Código vs Cafeína — Coloreado por Éxito de Tarea\n"
        "Hipótesis: balance óptimo multivariado > efecto individual\n"
        "★ Zona Dorada = combinación de rangos con mayor tasa de éxito (3 variables)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Cafeína (coffee_intake_mg) — mg\n"
        "<200 mg = bajo (~<2 tazas)  |  200-400 mg = medio  |  >400 mg = alto (~>4 tazas)",
        fontsize=11,
    )
    ax.set_ylabel(
        "Horas de Código (hours_coding) — h/día\n"
        "<4 h = bajo  |  4-7 h = medio  |  >7 h = alto",
        fontsize=11,
    )
    ax.legend(title="Resultado de Tarea", fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "plan8_scatter_zona_dorada.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Guardado: {ruta}")


def guardar_score_compuesto(df: pd.DataFrame, corr_score: float) -> None:
    """
    Violinplot + boxplot del score compuesto separado por task_success.
    Anota medianas y muestra la correlación con las individuales en el título.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    paleta = {"0": "#e74c3c", "1": "#2ecc71"}

    # ── Panel izquierdo: violinplot ──
    ax_v = axes[0]
    sns.violinplot(
        data=df,
        x="task_success",
        y="score_compuesto",
        hue="task_success",
        palette=paleta,
        inner="box",
        legend=False,
        ax=ax_v,
    )
    ax_v.set_title(
        "Distribución del Score Compuesto por Resultado\n"
        "Score = 0.695·cafeína + 0.616·horas + 0.187·sueño (min-max normalizado)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_v.set_xlabel("Resultado de Tarea (0=Fracaso, 1=Éxito)", fontsize=11)
    ax_v.set_ylabel("Score Compuesto (0-1)", fontsize=11)
    ax_v.set_xticklabels(["Fracaso (0)", "Éxito (1)"], fontsize=11)

    # Anotar mediana y media de cada grupo
    for pos, resultado in enumerate([0, 1]):
        subset = df[df["task_success"] == resultado]
        mediana = subset["score_compuesto"].median()
        media = subset["score_compuesto"].mean()
        ax_v.annotate(
            f"Med: {mediana:.3f}\nAvg: {media:.3f}",
            xy=(pos, mediana),
            xytext=(pos + 0.32, mediana),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )

    # ── Panel derecho: boxplot + puntos individuales ──
    ax_b = axes[1]
    sns.boxplot(
        data=df,
        x="task_success",
        y="score_compuesto",
        hue="task_success",
        palette=paleta,
        legend=False,
        ax=ax_b,
        width=0.45,
    )
    sns.stripplot(
        data=df,
        x="task_success",
        y="score_compuesto",
        hue="task_success",
        palette=paleta,
        alpha=0.28,
        size=3,
        legend=False,
        ax=ax_b,
    )

    # Línea horizontal: mediana global como referencia
    med_global = df["score_compuesto"].median()
    ax_b.axhline(
        y=med_global,
        color="black",
        linestyle="--",
        alpha=0.55,
        linewidth=1.5,
        label=f"Mediana global: {med_global:.3f}",
    )
    ax_b.legend(fontsize=9)

    ax_b.set_title(
        f"Boxplot + Puntos  |  Correlación score vs task_success: r = {corr_score:.4f}\n"
        f"Comparativa: cafeína=0.695  horas=0.616  sueño=0.187  |  score={corr_score:.4f}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_b.set_xlabel("Resultado de Tarea (0=Fracaso, 1=Éxito)", fontsize=11)
    ax_b.set_ylabel("Score Compuesto (0-1)", fontsize=11)
    ax_b.set_xticklabels(["Fracaso (0)", "Éxito (1)"], fontsize=11)

    # Título general
    fig.suptitle(
        "Score Compuesto Ponderado — Plan 8: Balance Óptimo Multivariado\n"
        "Hipótesis: score combinado > correlación de cualquier variable individual",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "plan8_score_compuesto.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Guardado: {ruta}")


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    print("🧪 Iniciando Plan 8: Balance Óptimo Multivariado")
    print("=" * 65)

    # Crear directorio de salida (idempotente)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── Paso 1: Sesión Spark ─────────────────────────────────────────────────
    print("\n📊 Paso 1: Configuración de SparkSession")
    spark = (
        SparkSession.builder.appName("Balance_Optimo_Plan8")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Manejo robusto de ruta al dataset
    data_path = "../data/ai_dev_productivity.csv"
    if not os.path.exists(data_path):
        data_path = (
            "/Users/andrestamez5/Personal/BigDataProject/data/ai_dev_productivity.csv"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "No se encontró el dataset en ninguna ruta conocida. "
            "Verificar: ../data/ai_dev_productivity.csv"
        )

    print(f"📁 Dataset: {data_path}")
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)
    n_total = df_spark.count()
    print(f"✅ Dataset cargado: {n_total} registros, {len(df_spark.columns)} variables")

    # ─── Paso 2: Rangos en PySpark ────────────────────────────────────────────
    print("\n📈 Paso 2: Creando columnas de rango en PySpark")
    df_rangos_spark = agregar_rangos_spark(df_spark)

    # Distribución por rango (diagnóstico en consola)
    print("\n  🔍 Distribución cafeína_rango:")
    df_rangos_spark.groupBy("cafeina_rango").agg(
        count("*").alias("total"),
        avg("task_success").alias("tasa_exito"),
    ).orderBy("tasa_exito", ascending=False).show()

    print("  🔍 Distribución horas_rango:")
    df_rangos_spark.groupBy("horas_rango").agg(
        count("*").alias("total"),
        avg("task_success").alias("tasa_exito"),
    ).orderBy("tasa_exito", ascending=False).show()

    print("  🔍 Distribución sueno_rango:")
    df_rangos_spark.groupBy("sueno_rango").agg(
        count("*").alias("total"),
        avg("task_success").alias("tasa_exito"),
    ).orderBy("tasa_exito", ascending=False).show()

    # ─── Paso 3: Convertir a Pandas — luego detener Spark ────────────────────
    print("🔄 Paso 3: Convirtiendo a pandas y deteniendo Spark")
    df = df_rangos_spark.toPandas()

    # Detener Spark ANTES de usar matplotlib / seaborn
    spark.stop()
    print("✅ Sesión Spark detenida")

    # ─── Paso 4: Score compuesto ──────────────────────────────────────────────
    print("\n🧮 Paso 4: Calculando score compuesto ponderado")
    df = calcular_score_compuesto(df)

    corr_score = df["score_compuesto"].corr(df["task_success"])
    print(f"  Correlación score_compuesto vs task_success : r = {corr_score:.4f}")
    print(f"  Correlaciones individuales conocidas:")
    print(f"    coffee_intake_mg → {CORR_CAFEINA}")
    print(f"    hours_coding     → {CORR_HORAS}")
    print(f"    sleep_hours      → {CORR_SUENO}")

    stats_score = {}
    for resultado in [0, 1]:
        grupo = "ÉXITO" if resultado == 1 else "FRACASO"
        subset = df[df["task_success"] == resultado]["score_compuesto"]
        stats_score[resultado] = {
            "mean": subset.mean(),
            "median": subset.median(),
            "std": subset.std(),
            "min": subset.min(),
            "max": subset.max(),
            "n": len(subset),
        }
        print(f"\n  Score — {grupo}:")
        print(f"    Media:   {stats_score[resultado]['mean']:.4f}")
        print(f"    Mediana: {stats_score[resultado]['median']:.4f}")
        print(f"    Std:     {stats_score[resultado]['std']:.4f}")
        print(f"    N:       {stats_score[resultado]['n']}")

    # ─── Paso 5: Pivot tables ─────────────────────────────────────────────────
    print("\n📊 Paso 5: Calculando pivot tables para heatmaps")

    p_caf_horas = pivot_tasa(df, "cafeina_rango", "horas_rango")
    p_caf_sueno = pivot_tasa(df, "cafeina_rango", "sueno_rango")
    p_horas_sueno = pivot_tasa(df, "horas_rango", "sueno_rango")

    cnt_caf_horas = pivot_conteos(df, "cafeina_rango", "horas_rango")
    cnt_caf_sueno = pivot_conteos(df, "cafeina_rango", "sueno_rango")
    cnt_horas_sueno = pivot_conteos(df, "horas_rango", "sueno_rango")

    print(f"\n  Pivot cafeína × horas (tasa de éxito):\n{p_caf_horas.round(3)}")
    print(f"\n  Pivot cafeína × sueño (tasa de éxito):\n{p_caf_sueno.round(3)}")
    print(f"\n  Pivot horas   × sueño (tasa de éxito):\n{p_horas_sueno.round(3)}")

    # ─── Paso 6: Zona dorada y ranking ───────────────────────────────────────
    print("\n🌟 Paso 6: Identificando zona dorada (combinación triple)")

    ranking = (
        df.groupby(["cafeina_rango", "horas_rango", "sueno_rango"])
        .agg(
            tasa_exito=("task_success", "mean"),
            total=("task_success", "count"),
        )
        .reset_index()
        .sort_values("tasa_exito", ascending=False)
        .reset_index(drop=True)
    )

    print("\n  Top 10 combinaciones por tasa de éxito:")
    print(ranking.head(10).to_string(index=False))

    # Zona dorada: mejor combinación triple
    top = ranking.iloc[0]
    zona_optima = {
        "cafeina_rango": top["cafeina_rango"],
        "horas_rango": top["horas_rango"],
        "sueno_rango": top["sueno_rango"],
        "tasa": top["tasa_exito"],
        "n": int(top["total"]),
    }

    # Zona destructora: peor combinación
    bottom = ranking.iloc[-1]
    zona_peor = {
        "cafeina_rango": bottom["cafeina_rango"],
        "horas_rango": bottom["horas_rango"],
        "sueno_rango": bottom["sueno_rango"],
        "tasa": bottom["tasa_exito"],
        "n": int(bottom["total"]),
    }

    print(
        f"\n  ★  ZONA DORADA    → cafeína={zona_optima['cafeina_rango']} | "
        f"horas={zona_optima['horas_rango']} | sueño={zona_optima['sueno_rango']} "
        f"→ {zona_optima['tasa'] * 100:.1f}%  (n={zona_optima['n']})"
    )
    print(
        f"  ✗  ZONA DESTRUCTORA → cafeína={zona_peor['cafeina_rango']} | "
        f"horas={zona_peor['horas_rango']} | sueño={zona_peor['sueno_rango']} "
        f"→ {zona_peor['tasa'] * 100:.1f}%  (n={zona_peor['n']})"
    )

    # ─── Paso 7: Visualizaciones ─────────────────────────────────────────────
    print("\n🎨 Paso 7: Generando visualizaciones (5 gráficos)")
    plt.style.use("default")
    sns.set_palette("husl")

    # Gráfico 1 — heatmap cafeína × horas
    print("\n  📈 Gráfico 1: Heatmap cafeína × horas")
    guardar_heatmap_pares(
        p_caf_horas,
        cnt_caf_horas,
        "Cafeína",
        "Horas de Código",
        "plan8_heatmap_cafeina_horas.png",
        alias_y=ALIAS_CAFEINA,
        alias_x=ALIAS_HORAS,
    )

    # Gráfico 2 — heatmap cafeína × sueño
    print("  📈 Gráfico 2: Heatmap cafeína × sueño")
    guardar_heatmap_pares(
        p_caf_sueno,
        cnt_caf_sueno,
        "Cafeína",
        "Sueño",
        "plan8_heatmap_cafeina_sueno.png",
        alias_y=ALIAS_CAFEINA,
        alias_x=ALIAS_SUENO,
    )

    # Gráfico 3 — heatmap horas × sueño
    print("  📈 Gráfico 3: Heatmap horas × sueño")
    guardar_heatmap_pares(
        p_horas_sueno,
        cnt_horas_sueno,
        "Horas de Código",
        "Sueño",
        "plan8_heatmap_horas_sueno.png",
        alias_y=ALIAS_HORAS,
        alias_x=ALIAS_SUENO,
    )

    # Gráfico 4 — scatter zona dorada
    print("  📈 Gráfico 4: Scatter zona dorada")
    guardar_scatter_zona_dorada(df, zona_optima)

    # Gráfico 5 — score compuesto
    print("  📈 Gráfico 5: Score compuesto (violin + boxplot)")
    guardar_score_compuesto(df, corr_score)

    # ─── Paso 8: Archivo de estadísticas ─────────────────────────────────────
    print("\n📋 Paso 8: Guardando archivo de estadísticas")
    stats_path = os.path.join(OUTPUT_DIR, "plan8_balance_estadisticas.txt")

    mejora_vs_cafeina = corr_score - CORR_CAFEINA  # diferencia vs mejor individual

    with open(stats_path, "w", encoding="utf-8") as f:
        # ── Sección 1 ──
        f.write("=== PLAN 8: BALANCE ÓPTIMO CAFEÍNA + HORAS + SUEÑO ===\n")
        f.write(
            'Hipótesis: "Existe un balance óptimo entre las tres variables '
            'que maximiza task_success"\n'
        )
        f.write(
            f"Correlaciones individuales conocidas: "
            f"cafeína={CORR_CAFEINA}, horas={CORR_HORAS}, sueño={CORR_SUENO}\n\n"
        )

        # ── Sección 2 ──
        f.write("=== METODOLOGÍA ===\n")
        f.write("• Dataset: 500 registros de productividad de desarrolladores\n")
        f.write("• Variables combinadas: coffee_intake_mg, hours_coding, sleep_hours\n")
        f.write("• Variable objetivo: task_success (binaria: 0=fracaso, 1=éxito)\n")
        f.write(
            "• Enfoque 1 — Score compuesto: normalización min-max + ponderación por correlaciones\n"
        )
        f.write(
            "• Enfoque 2 — Heatmaps de pares: pivot_table con tasa media de éxito\n"
        )
        f.write(
            "• Enfoque 3 — Ranking triple: groupby (cafeina_rango, horas_rango, sueno_rango)\n"
        )
        f.write(
            "• Herramientas: PySpark (rangos + distribuciones), Pandas/Seaborn (pivots + viz)\n\n"
        )

        # ── Sección 3 ──
        f.write("=== DEFINICIÓN DE RANGOS ===\n")
        f.write("Cafeína (coffee_intake_mg):\n")
        f.write("  • BAJO:  < 200 mg   (~menos de 2 tazas de café, 95 mg/taza)\n")
        f.write("  • MEDIO: 200–400 mg (~2 a 4 tazas de café)\n")
        f.write("  • ALTO:  > 400 mg   (~más de 4 tazas de café)\n")
        f.write("Horas de Código (hours_coding):\n")
        f.write("  • BAJO:  < 4 h/día\n")
        f.write("  • MEDIO: 4–7 h/día\n")
        f.write("  • ALTO:  > 7 h/día\n")
        f.write("Sueño (sleep_hours):\n")
        f.write("  • BAJO:  < 6 h/noche\n")
        f.write("  • MEDIO: 6–8 h/noche (rango recomendado OMS para adultos)\n")
        f.write("  • ALTO:  > 8 h/noche\n\n")

        # ── Sección 4 ──
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        sc_global = df["score_compuesto"]
        f.write("Score compuesto GLOBAL:\n")
        f.write(f"  • Media:   {sc_global.mean():.4f}\n")
        f.write(f"  • Mediana: {sc_global.median():.4f}\n")
        f.write(f"  • Std:     {sc_global.std():.4f}\n")
        f.write(f"  • Min:     {sc_global.min():.4f}\n")
        f.write(f"  • Max:     {sc_global.max():.4f}\n\n")

        for resultado in [1, 0]:
            grupo_label = (
                "ÉXITO (task_success=1)"
                if resultado == 1
                else "FRACASO (task_success=0)"
            )
            s = stats_score[resultado]
            f.write(f"Score compuesto — Grupo {grupo_label}:\n")
            f.write(f"  • Media:   {s['mean']:.4f}\n")
            f.write(f"  • Mediana: {s['median']:.4f}\n")
            f.write(f"  • Std:     {s['std']:.4f}\n")
            f.write(f"  • Min:     {s['min']:.4f}\n")
            f.write(f"  • Max:     {s['max']:.4f}\n")
            f.write(f"  • N:       {s['n']}\n\n")

        f.write(
            f"Correlación score_compuesto vs task_success: r = {corr_score:.4f}\n\n"
        )

        # ── Sección 5 ──
        f.write("=== ANÁLISIS POR COMBINACIONES ===\n")
        f.write("Top 10 combinaciones de rangos por tasa de éxito:\n\n")
        for rank, row in ranking.head(10).iterrows():
            f.write(
                f"  {rank + 1:2d}. cafeína={row['cafeina_rango'].upper():5s}  "
                f"horas={row['horas_rango'].upper():5s}  "
                f"sueño={row['sueno_rango'].upper():5s}  →  "
                f"{row['tasa_exito'] * 100:.1f}%  (n={int(row['total'])})\n"
            )
        f.write("\nPeores 5 combinaciones:\n")
        for rank, row in ranking.tail(5).iterrows():
            f.write(
                f"  cafeína={row['cafeina_rango'].upper():5s}  "
                f"horas={row['horas_rango'].upper():5s}  "
                f"sueño={row['sueno_rango'].upper():5s}  →  "
                f"{row['tasa_exito'] * 100:.1f}%  (n={int(row['total'])})\n"
            )
        f.write("\n")

        # Heatmaps resumidos en texto
        f.write("Pivot cafeína × horas (tasa de éxito media):\n")
        f.write(p_caf_horas.round(3).to_string())
        f.write("\n\nPivot cafeína × sueño (tasa de éxito media):\n")
        f.write(p_caf_sueno.round(3).to_string())
        f.write("\n\nPivot horas × sueño (tasa de éxito media):\n")
        f.write(p_horas_sueno.round(3).to_string())
        f.write("\n\n")

        # ── Sección 6 ──
        f.write("=== INSIGHTS CLAVE ===\n")
        f.write("ZONA DORADA (combinación que maximiza task_success):\n")
        f.write(f"  • Cafeína: {zona_optima['cafeina_rango'].upper()}\n")
        f.write(f"  • Horas de código: {zona_optima['horas_rango'].upper()}\n")
        f.write(f"  • Sueño: {zona_optima['sueno_rango'].upper()}\n")
        f.write(f"  • Tasa de éxito: {zona_optima['tasa'] * 100:.1f}%\n")
        f.write(f"  • N registros en la zona: {zona_optima['n']}\n\n")

        f.write("ZONA DESTRUCTORA (combinación que minimiza task_success):\n")
        f.write(f"  • Cafeína: {zona_peor['cafeina_rango'].upper()}\n")
        f.write(f"  • Horas de código: {zona_peor['horas_rango'].upper()}\n")
        f.write(f"  • Sueño: {zona_peor['sueno_rango'].upper()}\n")
        f.write(f"  • Tasa de éxito: {zona_peor['tasa'] * 100:.1f}%\n")
        f.write(f"  • N registros en la zona: {zona_peor['n']}\n\n")

        # ── Sección 7 ──
        f.write("=== VEREDICTO FINAL ===\n")
        f.write("¿El score compuesto supera la correlación individual más fuerte?\n")
        f.write(f"  • Mejor correlación individual (cafeína): {CORR_CAFEINA}\n")
        f.write(f"  • Correlación score compuesto:            {corr_score:.4f}\n")
        f.write(
            f"  • Diferencia:                             {mejora_vs_cafeina:+.4f}\n\n"
        )

        if corr_score > CORR_CAFEINA:
            f.write(
                "✅ HIPÓTESIS CONFIRMADA: el score compuesto SUPERA la mejor correlación\n"
                "   individual. La combinación multivariada es más predictiva que cualquier\n"
                "   variable sola.\n"
            )
        elif corr_score > CORR_HORAS:
            f.write(
                "🔄 HIPÓTESIS PARCIALMENTE CONFIRMADA: el score supera a horas y sueño\n"
                "   individualmente, pero no alcanza a cafeína sola. La combinación agrega\n"
                "   valor sobre las variables de menor correlación.\n"
            )
        else:
            f.write(
                "❌ HIPÓTESIS NO CONFIRMADA: el score compuesto lineal no supera las\n"
                "   correlaciones individuales. Posible razón: la ponderación lineal\n"
                "   no captura interacciones no lineales entre las variables.\n"
            )

        f.write(
            f"\nNivel del score: "
            f"{'Fuerte (>0.5)' if corr_score > 0.5 else 'Moderado (0.3-0.5)' if corr_score > 0.3 else 'Débil (<0.3)'}\n\n"
        )

        # ── Sección 8 ──
        lim_concretos = {
            "bajo": {
                "cafeina": "<200 mg (~<2 tazas)",
                "horas": "<4 h/día",
                "sueno": "<6 h/noche",
            },
            "medio": {
                "cafeina": "200-400 mg (~2-4 tazas)",
                "horas": "4-7 h/día",
                "sueno": "6-8 h/noche",
            },
            "alto": {
                "cafeina": ">400 mg (~>4 tazas)",
                "horas": ">7 h/día",
                "sueno": ">8 h/noche",
            },
        }
        c = zona_optima["cafeina_rango"]
        h = zona_optima["horas_rango"]
        s = zona_optima["sueno_rango"]

        f.write("=== RECOMENDACIONES PRÁCTICAS ===\n")
        f.write("Combinación óptima en términos concretos:\n")
        f.write(f"  • Cafeína:           {lim_concretos[c]['cafeina']}\n")
        f.write(f"  • Horas de código:   {lim_concretos[h]['horas']}\n")
        f.write(f"  • Sueño:             {lim_concretos[s]['sueno']}\n")
        f.write(
            f"  → Tasa de éxito esperada en este dataset: {zona_optima['tasa'] * 100:.1f}%\n\n"
        )
        f.write(
            "1. Perseguir la combinación óptima SIMULTANEAMENTE, no optimizar una variable sola.\n"
        )
        f.write(
            "2. El sueño (correlación individual baja=0.187) puede actuar como modulador;\n"
        )
        f.write(
            "   su efecto se magnifica en presencia de alta cafeína y muchas horas de código.\n"
        )
        f.write(
            "3. No compensar déficit de sueño con más cafeína: la zona dorada requiere\n"
        )
        f.write("   los tres factores en niveles adecuados de forma simultánea.\n")
        f.write(
            "4. Evitar la zona destructora: identifica la combinación de rangos con\n"
        )
        f.write("   menor tasa de éxito y alejarse de ella activamente.\n\n")

        # ── Sección 9 ──
        f.write("=== LIMITACIONES DEL ANÁLISIS ===\n")
        f.write("• Correlación no implica causalidad.\n")
        f.write(
            "• El score compuesto asume linealidad e independencia entre variables;\n"
        )
        f.write(
            "  posibles interacciones no lineales (synergias, umbrales) no son capturadas.\n"
        )
        f.write(
            "• Algunos rangos tienen n pequeño → menor confianza estadística en esas celdas.\n"
        )
        f.write(
            "• Los umbrales de rango son arbitrarios; umbrales alternativos pueden\n"
        )
        f.write("  cambiar los resultados de la zona dorada.\n")
        f.write(
            "• Dataset limitado a 500 registros; resultados pueden no generalizar.\n"
        )
        f.write(
            "• No se controlan variables confusoras (experiencia, tipo de tarea, stack).\n"
        )
        f.write(
            "• Para análisis más robusto se recomienda un modelo de regresión logística\n"
        )
        f.write(
            "  con términos de interacción o un árbol de decisión con las 3 variables.\n"
        )
        f.write(
            f"\nAnálisis completado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    print(f"    ✅ Estadísticas guardadas: {stats_path}")

    # ─── Resumen final en consola ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅ Plan 8 completado exitosamente")
    print(f"📊 Score compuesto vs task_success : r = {corr_score:.4f}")
    print(f"   Cafeína individual              : r = {CORR_CAFEINA}")
    print(f"   Diferencia                      : {mejora_vs_cafeina:+.4f}")
    print(
        f"\n★  ZONA DORADA: cafeína={zona_optima['cafeina_rango']} | "
        f"horas={zona_optima['horas_rango']} | sueño={zona_optima['sueno_rango']} "
        f"→ {zona_optima['tasa'] * 100:.1f}% éxito"
    )
    print(f"\n📁 Archivos en {OUTPUT_DIR}/")
    for nombre in [
        "plan8_heatmap_cafeina_horas.png",
        "plan8_heatmap_cafeina_sueno.png",
        "plan8_heatmap_horas_sueno.png",
        "plan8_scatter_zona_dorada.png",
        "plan8_score_compuesto.png",
        "plan8_balance_estadisticas.txt",
    ]:
        print(f"   - {nombre}")


if __name__ == "__main__":
    main()
