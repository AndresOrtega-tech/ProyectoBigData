#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan 7: Trade-off Commits vs Bugs Reportados — Análisis de Productividad vs Calidad
Hipótesis: "Existe un trade-off entre cantidad de commits y calidad del código (bugs reportados):
mayor cantidad de commits se asocia con más bugs, y ambos factores combinados afectan
negativamente a task_success."
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, expr, when
from pyspark.sql.functions import corr as spark_corr

# ── Función auxiliar ───────────────────────────────────────────────────────────


def _interpretar_corr(r: float) -> str:
    """Retorna descripción textual de la fuerza de una correlación de Pearson."""
    abs_r = abs(r)
    if abs_r < 0.05:
        fuerza = "nula"
    elif abs_r < 0.1:
        fuerza = "casi nula"
    elif abs_r < 0.3:
        fuerza = "débil"
    elif abs_r < 0.5:
        fuerza = "moderada"
    elif abs_r < 0.7:
        fuerza = "fuerte"
    else:
        fuerza = "muy fuerte"
    direccion = "positiva" if r >= 0 else "negativa"
    return f"  ← {direccion} {fuerza}"


def main():
    print("🧪 Iniciando Plan 7: Trade-off Commits vs Bugs")

    # ── Paso 1: Configuración inicial ──────────────────────────────────────────
    print("\n📊 Paso 1: Configuración inicial")

    spark = (
        SparkSession.builder.appName("Commits_Bugs_Tradeoff")
        .master("local[*]")
        .getOrCreate()
    )

    # Suprimir logs de Spark para salida más limpia
    spark.sparkContext.setLogLevel("ERROR")

    # Carga del dataset con fallback a ruta absoluta
    data_path = "../data/ai_dev_productivity.csv"
    if not os.path.exists(data_path):
        data_path = (
            "/Users/andrestamez5/Personal/BigDataProject/data/ai_dev_productivity.csv"
        )

    print(f"📁 Usando dataset: {data_path}")
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)
    total_registros = df_spark.count()
    print(f"✅ Dataset cargado: {total_registros} registros")

    # ── Paso 2: Análisis en PySpark ────────────────────────────────────────────
    print("\n📈 Paso 2: Análisis en PySpark")

    # 2.1 Correlaciones principales entre las tres variables del trade-off
    print("🔍 2.1 Calculando correlaciones")
    corr_commits_success = df_spark.stat.corr("commits", "task_success")
    corr_bugs_success = df_spark.stat.corr("bugs_reported", "task_success")
    corr_commits_bugs = df_spark.stat.corr("commits", "bugs_reported")

    print(f"  commits       vs task_success : {corr_commits_success:+.3f}")
    print(f"  bugs_reported vs task_success : {corr_bugs_success:+.3f}")
    print(f"  commits       vs bugs_reported: {corr_commits_bugs:+.3f}")

    # 2.2 Estadísticas descriptivas de percentiles para ambas variables
    print("🔍 2.2 Estadísticas descriptivas de commits y bugs")
    stats_commits = df_spark.select(
        expr("percentile_approx(commits, 0.5)").alias("mediana"),
        expr("percentile_approx(commits, 0.25)").alias("q1"),
        expr("percentile_approx(commits, 0.75)").alias("q3"),
        expr("avg(commits)").alias("promedio"),
        expr("min(commits)").alias("minimo"),
        expr("max(commits)").alias("maximo"),
    ).collect()[0]

    stats_bugs = df_spark.select(
        expr("percentile_approx(bugs_reported, 0.5)").alias("mediana"),
        expr("percentile_approx(bugs_reported, 0.25)").alias("q1"),
        expr("percentile_approx(bugs_reported, 0.75)").alias("q3"),
        expr("avg(bugs_reported)").alias("promedio"),
        expr("min(bugs_reported)").alias("minimo"),
        expr("max(bugs_reported)").alias("maximo"),
    ).collect()[0]

    print(
        f"  commits  — mediana: {stats_commits['mediana']}, "
        f"promedio: {stats_commits['promedio']:.2f}, "
        f"rango [{stats_commits['minimo']}, {stats_commits['maximo']}]"
    )
    print(
        f"  bugs     — mediana: {stats_bugs['mediana']}, "
        f"promedio: {stats_bugs['promedio']:.2f}, "
        f"rango [{stats_bugs['minimo']}, {stats_bugs['maximo']}]"
    )

    # 2.3 Creación de rangos discretos para commits y bugs
    # commits: bajo < 3 | medio 3-6 | alto > 6
    # bugs: sin_bugs = 0 | pocos = 1-2 | critico ≥ 3
    print("🔍 2.3 Creando rangos categóricos")
    df_con_rangos = df_spark.withColumn(
        "commits_rango",
        when(col("commits") < 3, "bajo")
        .when((col("commits") >= 3) & (col("commits") <= 6), "medio")
        .otherwise("alto"),
    ).withColumn(
        "bugs_rango",
        when(col("bugs_reported") == 0, "sin_bugs")
        .when((col("bugs_reported") >= 1) & (col("bugs_reported") <= 2), "pocos")
        .otherwise("critico"),
    )

    # 2.4 Tasa de éxito y promedio de bugs por rango de commits
    print("🔍 2.4 Tasa de éxito por rango de commits")
    tasa_commits = (
        df_con_rangos.groupBy("commits_rango")
        .agg(
            avg("task_success").alias("tasa_exito"),
            count("*").alias("total"),
            avg("bugs_reported").alias("avg_bugs"),
        )
        .orderBy("commits_rango")
    )

    tasa_commits.show()

    # 2.5 Tasa de éxito y promedio de commits por rango de bugs
    print("🔍 2.5 Tasa de éxito por rango de bugs")
    tasa_bugs = (
        df_con_rangos.groupBy("bugs_rango")
        .agg(
            avg("task_success").alias("tasa_exito"),
            count("*").alias("total"),
            avg("commits").alias("avg_commits"),
        )
        .orderBy("bugs_rango")
    )

    tasa_bugs.show()

    # 2.6 Análisis combinado: commits_rango × bugs_rango → tasa éxito (para heatmap)
    print("🔍 2.6 Tabla combinada commits_rango × bugs_rango")
    tasa_combinada = (
        df_con_rangos.groupBy("commits_rango", "bugs_rango")
        .agg(
            avg("task_success").alias("tasa_exito"),
            count("*").alias("total"),
        )
        .orderBy("commits_rango", "bugs_rango")
    )

    tasa_combinada.show()

    # 2.7 Distribución de registros por rango (para reporte de proporciones)
    print("🔍 2.7 Distribución por rango")
    conteo_commits_rango = (
        df_con_rangos.groupBy("commits_rango")
        .agg(count("*").alias("frecuencia"))
        .collect()
    )
    conteo_bugs_rango = (
        df_con_rangos.groupBy("bugs_rango")
        .agg(count("*").alias("frecuencia"))
        .collect()
    )

    for row in sorted(conteo_commits_rango, key=lambda r: r["commits_rango"]):
        pct = row["frecuencia"] / total_registros * 100
        print(
            f"  commits [{row['commits_rango']:6s}]: {row['frecuencia']} registros ({pct:.1f}%)"
        )

    for row in sorted(conteo_bugs_rango, key=lambda r: r["bugs_rango"]):
        pct = row["frecuencia"] / total_registros * 100
        print(
            f"  bugs    [{row['bugs_rango']:10s}]: {row['frecuencia']} registros ({pct:.1f}%)"
        )

    # ── Paso 3: Convertir a Pandas y cerrar Spark ──────────────────────────────
    print("\n🔄 Paso 3: Convirtiendo a Pandas y cerrando Spark")

    df_pd = df_spark.toPandas()
    tasa_commits_pd = tasa_commits.toPandas()
    tasa_bugs_pd = tasa_bugs.toPandas()
    tasa_combinada_pd = tasa_combinada.toPandas()

    spark.stop()
    print("✅ Spark detenido correctamente")

    # ── Paso 4: Visualizaciones ────────────────────────────────────────────────
    print("\n📊 Paso 4: Generando visualizaciones")

    # Crear carpeta de salida
    output_dir = "notebooks/results/plan7-commits-bugs"
    os.makedirs(output_dir, exist_ok=True)

    # Ordenes canonicos para los rangos
    orden_commits = ["bajo", "medio", "alto"]
    orden_bugs = ["sin_bugs", "pocos", "critico"]

    # Ordenar dataframes según el orden definido antes de graficar
    tasa_commits_pd["commits_rango"] = pd.Categorical(
        tasa_commits_pd["commits_rango"], categories=orden_commits, ordered=True
    )
    tasa_commits_pd = tasa_commits_pd.sort_values("commits_rango").reset_index(
        drop=True
    )

    tasa_bugs_pd["bugs_rango"] = pd.Categorical(
        tasa_bugs_pd["bugs_rango"], categories=orden_bugs, ordered=True
    )
    tasa_bugs_pd = tasa_bugs_pd.sort_values("bugs_rango").reset_index(drop=True)

    # Tasa de éxito global (referencia para todas las visualizaciones)
    tasa_global = df_pd["task_success"].mean()

    # ── Gráfico 1: Scatter commits vs bugs_reported coloreado por task_success ──
    print("📈 Gráfico 1: Scatter commits vs bugs_reported")
    fig, ax = plt.subplots(figsize=(12, 8))

    colores_success = {0: "#e74c3c", 1: "#2ecc71"}
    labels_success = {0: "Fracaso (task_success=0)", 1: "Éxito (task_success=1)"}

    for success_val in [0, 1]:
        subset = df_pd[df_pd["task_success"] == success_val]
        ax.scatter(
            subset["commits"],
            subset["bugs_reported"],
            c=colores_success[success_val],
            label=labels_success[success_val],
            alpha=0.60,
            edgecolors="white",
            linewidths=0.5,
            s=70,
            zorder=3,
        )

    # Línea de tendencia general (regresión lineal sobre todos los puntos)
    z = np.polyfit(df_pd["commits"], df_pd["bugs_reported"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_pd["commits"].min(), df_pd["commits"].max(), 200)
    ax.plot(
        x_range,
        p(x_range),
        "k--",
        alpha=0.7,
        linewidth=2.0,
        label=f"Tendencia (pendiente={z[0]:.3f})",
        zorder=4,
    )

    # Líneas de referencia para los umbrales de rango de commits
    ax.axvline(
        x=3,
        color="#f39c12",
        linestyle="--",
        alpha=0.85,
        linewidth=1.8,
        label="Umbral bajo→medio (commits=3)",
        zorder=2,
    )
    ax.axvline(
        x=6,
        color="#e67e22",
        linestyle="--",
        alpha=0.85,
        linewidth=1.8,
        label="Umbral medio→alto (commits=6)",
        zorder=2,
    )

    # Línea de referencia para el umbral de calidad de bugs
    ax.axhline(
        y=2,
        color="#9b59b6",
        linestyle=":",
        alpha=0.85,
        linewidth=1.8,
        label="Umbral pocos→crítico (bugs=2)",
        zorder=2,
    )

    # Anotación de correlaciones en el gráfico
    ax.annotate(
        f"Correlaciones de Pearson:\n"
        f"  r(commits, bugs)    = {corr_commits_bugs:+.3f}\n"
        f"  r(commits, success) = {corr_commits_success:+.3f}\n"
        f"  r(bugs,    success) = {corr_bugs_success:+.3f}",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#ecf0f1",
            alpha=0.90,
            edgecolor="#bdc3c7",
        ),
    )

    # Etiquetas de zonas de commits en el eje X
    y_label_pos = df_pd["bugs_reported"].max() * 1.05
    ax.text(
        1.5,
        y_label_pos,
        "BAJO\n(<3)",
        color="#f39c12",
        fontsize=9,
        ha="center",
        style="italic",
        fontweight="bold",
    )
    ax.text(
        4.5,
        y_label_pos,
        "MEDIO\n(3–6)",
        color="#e67e22",
        fontsize=9,
        ha="center",
        style="italic",
        fontweight="bold",
    )
    ax.text(
        8.0,
        y_label_pos,
        "ALTO\n(>6)",
        color="#c0392b",
        fontsize=9,
        ha="center",
        style="italic",
        fontweight="bold",
    )

    ax.set_title(
        "Trade-off Commits vs Bugs Reportados\n"
        "Hipótesis: Más commits → más bugs → menor éxito",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Commits (cantidad de commits por sesión)", fontsize=12)
    ax.set_ylabel("Bugs Reportados (cantidad de bugs detectados)", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/plan7_commits_bugs_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  ✅ Gráfico 1 guardado")

    # ── Gráfico 2: Barplot tasa de éxito por commits_rango ────────────────────
    print("📈 Gráfico 2: Tasa de éxito por rango de commits")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Verde→naranja→rojo para indicar calidad decreciente al aumentar commits
    colores_commits = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(
        range(len(tasa_commits_pd)),
        tasa_commits_pd["tasa_exito"],
        color=colores_commits[: len(tasa_commits_pd)],
        edgecolor="white",
        linewidth=1.5,
        width=0.55,
    )

    # Etiquetas sobre cada barra: % de éxito + n de registros
    for bar, (_, row) in zip(bars, tasa_commits_pd.iterrows()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.014,
            f"{height:.1%}\n(n={int(row['total'])})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        # Promedio de bugs reportados dentro de la barra
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 0.50,
            f"avg bugs\n{row['avg_bugs']:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    # Línea de referencia: tasa global del dataset
    ax.axhline(
        y=tasa_global,
        color="#2c3e50",
        linestyle="--",
        alpha=0.80,
        linewidth=1.8,
        label=f"Media global: {tasa_global:.1%}",
    )

    ax.set_title(
        "Tasa de Éxito por Rango de Commits\n¿Commitear más implica más o menos éxito?",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Rango de Commits", fontsize=12)
    ax.set_ylabel("Tasa de Éxito (task_success = 1)", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(range(len(tasa_commits_pd)))
    ax.set_xticklabels(
        ["Bajo\n(< 3 commits)", "Medio\n(3–6 commits)", "Alto\n(> 6 commits)"],
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/plan7_commits_bugs_tasa_exito_commits.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  ✅ Gráfico 2 guardado")

    # ── Gráfico 3: Barplot tasa de éxito por bugs_rango ───────────────────────
    print("📈 Gráfico 3: Tasa de éxito por rango de bugs")

    fig, ax = plt.subplots(figsize=(10, 7))

    colores_bugs = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(
        range(len(tasa_bugs_pd)),
        tasa_bugs_pd["tasa_exito"],
        color=colores_bugs[: len(tasa_bugs_pd)],
        edgecolor="white",
        linewidth=1.5,
        width=0.55,
    )

    # Etiquetas sobre cada barra: % de éxito + n de registros
    for bar, (_, row) in zip(bars, tasa_bugs_pd.iterrows()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.014,
            f"{height:.1%}\n(n={int(row['total'])})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        # Promedio de commits dentro de la barra
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 0.50,
            f"avg commits\n{row['avg_commits']:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    ax.axhline(
        y=tasa_global,
        color="#2c3e50",
        linestyle="--",
        alpha=0.80,
        linewidth=1.8,
        label=f"Media global: {tasa_global:.1%}",
    )

    ax.set_title(
        "Tasa de Éxito por Rango de Bugs Reportados\n"
        "¿Más bugs = menor probabilidad de completar la tarea?",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Rango de Bugs Reportados", fontsize=12)
    ax.set_ylabel("Tasa de Éxito (task_success = 1)", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(range(len(tasa_bugs_pd)))
    ax.set_xticklabels(
        ["Sin Bugs\n(0 bugs)", "Pocos\n(1–2 bugs)", "Crítico\n(3+ bugs)"],
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/plan7_commits_bugs_tasa_exito_bugs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  ✅ Gráfico 3 guardado")

    # ── Gráfico 4: Heatmap combinado commits_rango × bugs_rango ───────────────
    print("📈 Gráfico 4: Heatmap combinado commits × bugs")

    # Construir pivots para tasa de éxito y conteo de registros
    pivot_tasa = tasa_combinada_pd.pivot(
        index="bugs_rango",
        columns="commits_rango",
        values="tasa_exito",
    )
    pivot_conteo = tasa_combinada_pd.pivot(
        index="bugs_rango",
        columns="commits_rango",
        values="total",
    )

    # Reordenar filas y columnas según orden canónico
    pivot_tasa = pivot_tasa.reindex(index=orden_bugs, columns=orden_commits)
    pivot_conteo = pivot_conteo.reindex(index=orden_bugs, columns=orden_commits)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Heatmap: verde (alto éxito) → amarillo → rojo (bajo éxito)
    sns.heatmap(
        pivot_tasa,
        annot=False,  # anotaciones manuales para incluir conteo
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Tasa de Éxito (task_success)", "shrink": 0.80},
    )

    # Anotaciones manuales en cada celda: tasa + conteo de registros
    for i, bugs_cat in enumerate(orden_bugs):
        for j, commits_cat in enumerate(orden_commits):
            tasa_val = pivot_tasa.loc[bugs_cat, commits_cat]
            conteo_val = pivot_conteo.loc[bugs_cat, commits_cat]
            if pd.notna(tasa_val):
                # Color de texto según contraste con el fondo
                color_texto = (
                    "white" if (tasa_val < 0.35 or tasa_val > 0.72) else "black"
                )
                ax.text(
                    j + 0.50,
                    i + 0.50,
                    f"{tasa_val:.1%}\n(n={int(conteo_val)})",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=color_texto,
                )

    ax.set_title(
        "Heatmap: Combinación Commits × Bugs → Tasa de Éxito\n"
        "Verde = alta tasa de éxito   |   Rojo = baja tasa de éxito",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Rango de Commits", fontsize=12, labelpad=10)
    ax.set_ylabel("Rango de Bugs Reportados", fontsize=12, labelpad=10)
    ax.set_xticklabels(
        ["Bajo\n(< 3 commits)", "Medio\n(3–6 commits)", "Alto\n(> 6 commits)"],
        rotation=0,
        fontsize=11,
    )
    ax.set_yticklabels(
        ["Sin Bugs\n(= 0)", "Pocos\n(1–2)", "Crítico\n(≥ 3)"],
        rotation=0,
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/plan7_commits_bugs_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  ✅ Gráfico 4 guardado")

    # ── Paso 5: Archivo de estadísticas ───────────────────────────────────────
    print("\n📋 Paso 5: Generando archivo de estadísticas")

    # Métricas auxiliares para el informe
    commits_median = df_pd["commits"].median()
    bugs_median = df_pd["bugs_reported"].median()
    commits_mean = df_pd["commits"].mean()
    bugs_mean = df_pd["bugs_reported"].mean()
    commits_std = df_pd["commits"].std()
    bugs_std = df_pd["bugs_reported"].std()

    grupo_exito = df_pd[df_pd["task_success"] == 1]
    grupo_fracaso = df_pd[df_pd["task_success"] == 0]

    # Combinación óptima y peor según heatmap
    idx_optima = tasa_combinada_pd["tasa_exito"].idxmax()
    combinacion_optima = tasa_combinada_pd.loc[idx_optima]
    idx_peor = tasa_combinada_pd["tasa_exito"].idxmin()
    combinacion_peor = tasa_combinada_pd.loc[idx_peor]

    SEP = "=" * 65

    stats_path = f"{output_dir}/plan7_commits_bugs_estadisticas.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        # ── Encabezado ────────────────────────────────────────────────────────
        f.write(f"{SEP}\n")
        f.write("=== PLAN 7: TRADE-OFF COMMITS VS BUGS REPORTADOS ===\n")
        f.write(f"{SEP}\n")
        f.write(
            'Hipótesis: "A mayor cantidad de commits, mayor probabilidad de\n'
            "más bugs reportados, y ambos factores combinados afectan\n"
            'negativamente a task_success."\n\n'
        )

        # ── Sección 1: Metodología ────────────────────────────────────────────
        f.write(f"{SEP}\n=== METODOLOGÍA ===\n{SEP}\n")
        f.write(
            f"• Dataset: {total_registros} registros de productividad de desarrolladores\n"
        )
        f.write(
            "• Variable cuantitativa 1: commits (cantidad de commits por sesión de trabajo)\n"
        )
        f.write(
            "• Variable cuantitativa 2: bugs_reported (cantidad de bugs detectados en la sesión)\n"
        )
        f.write("• Variable dependiente: task_success (binaria: 0=fracaso, 1=éxito)\n")
        f.write("• Técnicas utilizadas:\n")
        f.write(
            "    – Correlación de Pearson (PySpark stat.corr) para cuantificar relaciones\n"
        )
        f.write(
            "    – Segmentación en rangos discretos mediante lógica condicional (when/otherwise)\n"
        )
        f.write("    – Agregación groupBy + avg para tasas de éxito por segmento\n")
        f.write(
            "    – Análisis matricial combinado (commits_rango × bugs_rango) vía heatmap\n"
        )
        f.write(
            "• Herramientas: PySpark 3.5, Pandas 3.0, Matplotlib 3.10, Seaborn 0.13\n\n"
        )

        # ── Sección 2: Definición de rangos ───────────────────────────────────
        f.write(f"{SEP}\n=== DEFINICIÓN DE RANGOS ===\n{SEP}\n")
        f.write("COMMITS (dimensión productividad / cantidad):\n")
        f.write(
            "  • bajo   : commits < 3     → poca actividad, sesión exploratoria o bloqueada\n"
        )
        f.write(
            "  • medio  : commits 3–6     → ritmo normal, flujo sostenido de trabajo\n"
        )
        f.write(
            "  • alto   : commits > 6     → alta actividad, posible presión o deuda técnica\n\n"
        )
        f.write("BUGS REPORTADOS (dimensión calidad):\n")
        f.write(
            "  • sin_bugs : bugs = 0      → código limpio, sin problemas detectados\n"
        )
        f.write(
            "  • pocos    : bugs = 1–2    → errores menores, manejables, fáciles de corregir\n"
        )
        f.write(
            "  • critico  : bugs ≥ 3      → múltiples problemas, riesgo para la entrega\n\n"
        )
        f.write("Justificación de umbrales:\n")
        f.write(
            f"  • Mediana commits: {commits_median:.0f}  → umbral medio centrado en distribución real\n"
        )
        f.write(
            f"  • Mediana bugs   : {bugs_median:.0f}  → umbral pocos/critico en zona de impacto\n\n"
        )

        # ── Sección 3: Estadísticas descriptivas ──────────────────────────────
        f.write(f"{SEP}\n=== ESTADÍSTICAS DESCRIPTIVAS ===\n{SEP}\n")
        f.write("COMMITS:\n")
        f.write(f"  • Promedio      : {commits_mean:.2f}\n")
        f.write(f"  • Mediana       : {commits_median:.1f}\n")
        f.write(f"  • Desv. Estándar: {commits_std:.2f}\n")
        f.write(
            f"  • Mínimo / Máximo: {df_pd['commits'].min():.0f} / {df_pd['commits'].max():.0f}\n"
        )
        f.write(f"  • Q1 / Q3       : {stats_commits['q1']} / {stats_commits['q3']}\n")
        f.write("  Por grupo de éxito:\n")
        f.write(
            f"    – Éxito   (n={len(grupo_exito):3d}): "
            f"prom={grupo_exito['commits'].mean():.2f}, "
            f"med={grupo_exito['commits'].median():.1f}, "
            f"std={grupo_exito['commits'].std():.2f}\n"
        )
        f.write(
            f"    – Fracaso (n={len(grupo_fracaso):3d}): "
            f"prom={grupo_fracaso['commits'].mean():.2f}, "
            f"med={grupo_fracaso['commits'].median():.1f}, "
            f"std={grupo_fracaso['commits'].std():.2f}\n\n"
        )

        f.write("BUGS REPORTADOS:\n")
        f.write(f"  • Promedio      : {bugs_mean:.2f}\n")
        f.write(f"  • Mediana       : {bugs_median:.1f}\n")
        f.write(f"  • Desv. Estándar: {bugs_std:.2f}\n")
        f.write(
            f"  • Mínimo / Máximo: {df_pd['bugs_reported'].min():.0f} / {df_pd['bugs_reported'].max():.0f}\n"
        )
        f.write(f"  • Q1 / Q3       : {stats_bugs['q1']} / {stats_bugs['q3']}\n")
        f.write("  Por grupo de éxito:\n")
        f.write(
            f"    – Éxito   (n={len(grupo_exito):3d}): "
            f"prom={grupo_exito['bugs_reported'].mean():.2f}, "
            f"med={grupo_exito['bugs_reported'].median():.1f}, "
            f"std={grupo_exito['bugs_reported'].std():.2f}\n"
        )
        f.write(
            f"    – Fracaso (n={len(grupo_fracaso):3d}): "
            f"prom={grupo_fracaso['bugs_reported'].mean():.2f}, "
            f"med={grupo_fracaso['bugs_reported'].median():.1f}, "
            f"std={grupo_fracaso['bugs_reported'].std():.2f}\n\n"
        )

        f.write("CORRELACIONES (Pearson):\n")
        f.write(
            f"  commits       vs task_success : {corr_commits_success:+.3f}"
            f"{_interpretar_corr(corr_commits_success)}\n"
        )
        f.write(
            f"  bugs_reported vs task_success : {corr_bugs_success:+.3f}"
            f"{_interpretar_corr(corr_bugs_success)}\n"
        )
        f.write(
            f"  commits       vs bugs_reported: {corr_commits_bugs:+.3f}"
            f"{_interpretar_corr(corr_commits_bugs)}\n\n"
        )

        # ── Sección 4: Análisis por categorías ────────────────────────────────
        f.write(f"{SEP}\n=== ANÁLISIS POR CATEGORÍAS ===\n{SEP}\n")

        f.write("TASA DE ÉXITO POR RANGO DE COMMITS:\n")
        for rango in orden_commits:
            subset = tasa_commits_pd[tasa_commits_pd["commits_rango"] == rango]
            if not subset.empty:
                row = subset.iloc[0]
                pct_ds = row["total"] / total_registros * 100
                f.write(
                    f"  {rango.upper():6s}: {row['tasa_exito']:.1%} éxito"
                    f" | n={int(row['total'])} ({pct_ds:.1f}% del dataset)"
                    f" | avg bugs reportados = {row['avg_bugs']:.2f}\n"
                )

        f.write("\nTASA DE ÉXITO POR RANGO DE BUGS:\n")
        for rango in orden_bugs:
            subset = tasa_bugs_pd[tasa_bugs_pd["bugs_rango"] == rango]
            if not subset.empty:
                row = subset.iloc[0]
                pct_ds = row["total"] / total_registros * 100
                f.write(
                    f"  {rango.upper():10s}: {row['tasa_exito']:.1%} éxito"
                    f" | n={int(row['total'])} ({pct_ds:.1f}% del dataset)"
                    f" | avg commits = {row['avg_commits']:.1f}\n"
                )

        f.write("\nHEATMAP COMBINADO — commits_rango × bugs_rango → tasa de éxito:\n")
        header = f"  {'':15s}| {'BAJO':>12s} | {'MEDIO':>12s} | {'ALTO':>12s}"
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        for bugs_cat in orden_bugs:
            row_values = []
            for commits_cat in orden_commits:
                val = (
                    pivot_tasa.loc[bugs_cat, commits_cat]
                    if bugs_cat in pivot_tasa.index
                    and commits_cat in pivot_tasa.columns
                    else float("nan")
                )
                row_values.append(f"{val:.1%}" if pd.notna(val) else "  N/A  ")
            f.write(
                f"  {bugs_cat.upper():15s}| {row_values[0]:>12s} | {row_values[1]:>12s} | {row_values[2]:>12s}\n"
            )

        f.write("\nDETALLE COMPLETO DE TODAS LAS COMBINACIONES:\n")
        tasa_combinada_sorted = tasa_combinada_pd.copy()
        tasa_combinada_sorted["cr_order"] = tasa_combinada_sorted["commits_rango"].map(
            {k: i for i, k in enumerate(orden_commits)}
        )
        tasa_combinada_sorted["br_order"] = tasa_combinada_sorted["bugs_rango"].map(
            {k: i for i, k in enumerate(orden_bugs)}
        )
        tasa_combinada_sorted = tasa_combinada_sorted.sort_values(
            ["cr_order", "br_order"]
        )
        for _, row in tasa_combinada_sorted.iterrows():
            f.write(
                f"  commits={row['commits_rango']:<6s} + bugs={row['bugs_rango']:<10s}"
                f" → {row['tasa_exito']:.1%} éxito  (n={int(row['total'])})\n"
            )
        f.write("\n")

        # ── Sección 5: Insights clave ──────────────────────────────────────────
        f.write(f"{SEP}\n=== INSIGHTS CLAVE ===\n{SEP}\n")

        # Insight 1: dirección del trade-off commits→bugs
        if corr_commits_bugs > 0.10:
            insight1 = (
                f"CONFIRMADO — correlación positiva ({corr_commits_bugs:+.3f}): "
                "más commits tienden a asociarse con más bugs"
            )
        elif corr_commits_bugs < -0.10:
            insight1 = (
                f"INVERTIDO — correlación negativa ({corr_commits_bugs:+.3f}): "
                "más commits se asocian con menos bugs"
            )
        else:
            insight1 = (
                f"AUSENTE — sin correlación significativa ({corr_commits_bugs:+.3f}): "
                "commits y bugs operan como variables independientes"
            )

        f.write(f"1. RELACIÓN COMMITS → BUGS:\n   {insight1}\n")
        f.write(
            f"   Diferencia en commits éxito vs fracaso: "
            f"{grupo_exito['commits'].mean():.2f} vs {grupo_fracaso['commits'].mean():.2f}\n\n"
        )

        # Insight 2: qué variable impacta más en el éxito
        if abs(corr_bugs_success) > abs(corr_commits_success):
            impacto_mayor = f"bugs_reported (r={corr_bugs_success:+.3f})"
            impacto_menor = f"commits (r={corr_commits_success:+.3f})"
        else:
            impacto_mayor = f"commits (r={corr_commits_success:+.3f})"
            impacto_menor = f"bugs_reported (r={corr_bugs_success:+.3f})"

        f.write(f"2. IMPACTO DIFERENCIAL EN TASK_SUCCESS:\n")
        f.write(f"   Mayor impacto : {impacto_mayor}\n")
        f.write(f"   Menor impacto : {impacto_menor}\n")
        f.write(f"   Tasa global de éxito del dataset: {tasa_global:.1%}\n\n")

        # Insight 3: combinación óptima
        f.write(f"3. COMBINACIÓN ÓPTIMA (mayor tasa de éxito):\n")
        f.write(
            f"   commits={combinacion_optima['commits_rango']} + bugs={combinacion_optima['bugs_rango']}\n"
        )
        f.write(
            f"   → Tasa de éxito: {combinacion_optima['tasa_exito']:.1%}  "
            f"(n={int(combinacion_optima['total'])})\n\n"
        )

        # Insight 4: combinación peor
        f.write(f"4. COMBINACIÓN PEOR (menor tasa de éxito):\n")
        f.write(
            f"   commits={combinacion_peor['commits_rango']} + bugs={combinacion_peor['bugs_rango']}\n"
        )
        f.write(
            f"   → Tasa de éxito: {combinacion_peor['tasa_exito']:.1%}  "
            f"(n={int(combinacion_peor['total'])})\n\n"
        )

        # Insight 5: spread entre mejor y peor combinación
        spread = combinacion_optima["tasa_exito"] - combinacion_peor["tasa_exito"]
        f.write(f"5. AMPLITUD DEL TRADE-OFF:\n")
        f.write(f"   Diferencia entre combinación óptima y peor: {spread:.1%}\n")
        f.write(
            f"   Interpretación: {'Alto impacto' if spread > 0.20 else 'Impacto moderado' if spread > 0.10 else 'Impacto bajo'} "
            f"de la combinación commits/bugs sobre el éxito\n\n"
        )

        # ── Sección 6: Veredicto final ─────────────────────────────────────────
        f.write(f"{SEP}\n=== VEREDICTO FINAL ===\n{SEP}\n")

        trade_off_commits_bugs = corr_commits_bugs > 0.05
        bugs_reducen_exito = corr_bugs_success < -0.05
        commits_reducen_exito = corr_commits_success < -0.05

        if trade_off_commits_bugs and bugs_reducen_exito:
            f.write("✅ TRADE-OFF PARCIALMENTE CONFIRMADO\n")
            f.write(
                "La cadena causal propuesta (commits → bugs → menor éxito) se sostiene:\n"
            )
            f.write(f"  • Más commits → más bugs     (r={corr_commits_bugs:+.3f})\n")
            f.write(f"  • Más bugs → menos éxito     (r={corr_bugs_success:+.3f})\n")
            f.write("Las correlaciones son débiles, lo que sugiere que el trade-off\n")
            f.write("existe pero no es el único determinante del éxito de la tarea.\n")
        elif not trade_off_commits_bugs and not bugs_reducen_exito:
            f.write("❌ TRADE-OFF NO CONFIRMADO\n")
            f.write("La evidencia no sostiene la hipótesis planteada:\n")
            f.write(
                f"  • commits vs bugs   : r={corr_commits_bugs:+.3f} (sin relación significativa)\n"
            )
            f.write(
                f"  • bugs vs éxito     : r={corr_bugs_success:+.3f} (sin relación significativa)\n"
            )
            f.write("Los commits y bugs operan como dimensiones independientes.\n")
            f.write(
                "El éxito de la tarea no está explicado principalmente por estas variables.\n"
            )
        else:
            f.write("⚠️  RESULTADO MIXTO — TRADE-OFF PARCIAL\n")
            f.write("La evidencia muestra patrones contradictorios o muy débiles:\n")
            f.write(f"  • commits vs bugs   : r={corr_commits_bugs:+.3f}\n")
            f.write(f"  • commits vs éxito  : r={corr_commits_success:+.3f}\n")
            f.write(f"  • bugs vs éxito     : r={corr_bugs_success:+.3f}\n")
            f.write("El análisis combinado del heatmap aporta más matices\n")
            f.write("que las correlaciones simples por separado.\n")

        abs_corr_max = max(abs(corr_commits_bugs), abs(corr_bugs_success))
        fuerza_tradeoff = (
            "DÉBIL"
            if abs_corr_max < 0.3
            else "MODERADO"
            if abs_corr_max < 0.5
            else "FUERTE"
        )
        f.write(f"\nFuerza general del trade-off observado: {fuerza_tradeoff}\n\n")

        # ── Sección 7: Recomendaciones prácticas ──────────────────────────────
        f.write(f"{SEP}\n=== RECOMENDACIONES PRÁCTICAS ===\n{SEP}\n")
        f.write(
            f"• Apuntar al rango de commits '{combinacion_optima['commits_rango']}' "
            f"asociado con la mayor tasa de éxito\n"
        )
        f.write(
            f"• Evitar la combinación crítica: commits={combinacion_peor['commits_rango']} "
            f"+ bugs={combinacion_peor['bugs_rango']} "
            f"(tasa mínima: {combinacion_peor['tasa_exito']:.1%})\n"
        )
        f.write(
            "• No sacrificar calidad (más bugs) para aumentar la cantidad de commits\n"
        )
        f.write(
            "• Si bugs_reported ≥ 3 (zona crítica), priorizar refactorización antes de commitear\n"
        )
        f.write(
            "• Implementar code review incremental para detectar bugs antes de acumularlos\n"
        )
        f.write(
            "• Usar el heatmap como mapa de riesgo: celdas rojas son señales de alerta operativa\n"
        )
        f.write(
            f"• Combinación recomendada: commits={combinacion_optima['commits_rango']} "
            f"+ bugs={combinacion_optima['bugs_rango']} "
            f"→ {combinacion_optima['tasa_exito']:.1%} de éxito esperado\n\n"
        )

        # ── Sección 8: Limitaciones ───────────────────────────────────────────
        f.write(f"{SEP}\n=== LIMITACIONES DEL ANÁLISIS ===\n{SEP}\n")
        f.write(
            "• Correlación no implica causalidad: factores externos no medidos pueden influir\n"
        )
        f.write(
            f"• Dataset limitado: {total_registros} registros pueden no generalizar a todos los contextos\n"
        )
        f.write("• No se controla por la complejidad inherente de cada tarea\n")
        f.write(
            "• El conteo de bugs depende de la cobertura del proceso de QA/testing\n"
        )
        f.write(
            "• La granularidad de commits varía por estilo (commits atómicos vs commits grandes)\n"
        )
        f.write(
            "• task_success es binaria y puede ocultar matices de rendimiento parcial\n"
        )
        f.write(
            "• Variables omitidas: pair programming, code review, experiencia del desarrollador\n"
        )
        f.write(
            "• Los umbrales de rango (3/6 commits, 2 bugs) son heurísticos — cambiarlos\n"
        )
        f.write("  podría modificar las conclusiones del análisis combinado\n")
        f.write("• El análisis de correlación de Pearson asume relación lineal;\n")
        f.write(
            "  relaciones no lineales (U-shape, threshold effects) podrían no captarse\n"
        )

    print(f"✅ Estadísticas guardadas en: {stats_path}")

    # ── Resumen final en consola ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("=== RESUMEN PLAN 7: TRADE-OFF COMMITS VS BUGS ===")
    print(f"{SEP}")
    print(
        f"  commits       vs task_success : {corr_commits_success:+.3f}"
        f"{_interpretar_corr(corr_commits_success)}"
    )
    print(
        f"  bugs_reported vs task_success : {corr_bugs_success:+.3f}"
        f"{_interpretar_corr(corr_bugs_success)}"
    )
    print(
        f"  commits       vs bugs_reported: {corr_commits_bugs:+.3f}"
        f"{_interpretar_corr(corr_commits_bugs)}"
    )
    print(f"\n  Tasa global de éxito: {tasa_global:.1%}")
    print(
        f"\n  Combinación óptima : commits={combinacion_optima['commits_rango']}"
        f" + bugs={combinacion_optima['bugs_rango']}"
        f" → {combinacion_optima['tasa_exito']:.1%}"
    )
    print(
        f"  Combinación peor   : commits={combinacion_peor['commits_rango']}"
        f" + bugs={combinacion_peor['bugs_rango']}"
        f" → {combinacion_peor['tasa_exito']:.1%}"
    )

    print("\n  Tasa de éxito por rango de commits:")
    for rango in orden_commits:
        subset = tasa_commits_pd[tasa_commits_pd["commits_rango"] == rango]
        if not subset.empty:
            row = subset.iloc[0]
            print(f"    {rango:<6s}: {row['tasa_exito']:.1%}  (n={int(row['total'])})")

    print("\n  Tasa de éxito por rango de bugs:")
    for rango in orden_bugs:
        subset = tasa_bugs_pd[tasa_bugs_pd["bugs_rango"] == rango]
        if not subset.empty:
            row = subset.iloc[0]
            print(f"    {rango:<10s}: {row['tasa_exito']:.1%}  (n={int(row['total'])})")

    print(f"\n✅ Plan 7 completado exitosamente")
    print(f"📁 Archivos guardados en {output_dir}/")
    print("   - plan7_commits_bugs_scatter.png")
    print("   - plan7_commits_bugs_tasa_exito_commits.png")
    print("   - plan7_commits_bugs_tasa_exito_bugs.png")
    print("   - plan7_commits_bugs_heatmap.png")
    print("   - plan7_commits_bugs_estadisticas.txt")


if __name__ == "__main__":
    main()
