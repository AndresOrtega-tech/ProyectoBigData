#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan 6: Hipótesis Uso de IA - Análisis de Correlación con Task Success
Determinar si mayor uso de herramientas de IA está asociado con mayor task_success.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, when


def main():
    print("🧪 Iniciando Plan 6: Análisis de Hipótesis Uso de IA")

    # Paso 1 — Configuración inicial
    print("\n📊 Paso 1: Configuración inicial")

    spark = (
        SparkSession.builder.appName("Hipotesis_Uso_IA")
        .master("local[*]")
        .getOrCreate()
    )

    # Cargar dataset con manejo robusto de rutas
    data_path = "../data/ai_dev_productivity.csv"
    if not os.path.exists(data_path):
        data_path = (
            "/Users/andrestamez5/Personal/BigDataProject/data/ai_dev_productivity.csv"
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "No se encuentra el dataset en ninguna ruta conocida"
            )

    print(f"📁 Usando dataset: {data_path}")
    df_spark = spark.read.csv(data_path, header=True, inferSchema=True)
    print(f"✅ Dataset cargado: {df_spark.count()} registros")

    # Paso 2 — Análisis en PySpark
    print("\n📈 Paso 2: Análisis en PySpark")

    # 2.1 Calcular correlación principal de Pearson
    print("🔍 2.1 Calcular correlación principal")
    correlacion = df_spark.stat.corr("ai_usage_hours", "task_success")
    print(f"Correlación ai_usage_hours vs task_success: {correlacion:.3f}")

    # 2.2 Correlaciones adicionales con otras variables
    print("🔍 2.2 Calcular correlaciones adicionales")
    correlacion_ia_commits = df_spark.stat.corr("ai_usage_hours", "commits")
    correlacion_ia_bugs = df_spark.stat.corr("ai_usage_hours", "bugs_reported")
    correlacion_ia_cognitiva = df_spark.stat.corr("ai_usage_hours", "cognitive_load")
    correlacion_ia_horas = df_spark.stat.corr("ai_usage_hours", "hours_coding")

    print(f"Correlación ai_usage_hours vs commits:       {correlacion_ia_commits:.3f}")
    print(f"Correlación ai_usage_hours vs bugs_reported: {correlacion_ia_bugs:.3f}")
    print(
        f"Correlación ai_usage_hours vs cognitive_load:{correlacion_ia_cognitiva:.3f}"
    )
    print(f"Correlación ai_usage_hours vs hours_coding:  {correlacion_ia_horas:.3f}")

    # 2.3 Promedio de uso de IA por task_success
    print("🔍 2.3 Promedio de uso de IA por task_success")
    promedio_ia = df_spark.groupBy("task_success").agg(
        avg("ai_usage_hours").alias("avg_ia_hours"),
        avg("hours_coding").alias("avg_coding"),
        avg("cognitive_load").alias("avg_cognitive_load"),
        avg("commits").alias("avg_commits"),
        count("*").alias("total_registros"),
    )
    promedio_ia.show()

    # 2.4 Crear rangos de ai_usage_hours: bajo < 2h, medio 2–4h, alto > 4h
    print("🔍 2.4 Crear rangos de uso de IA")
    df_con_rangos = df_spark.withColumn(
        "ia_rango",
        when(col("ai_usage_hours") < 2, "bajo")
        .when((col("ai_usage_hours") >= 2) & (col("ai_usage_hours") <= 4), "medio")
        .otherwise("alto"),
    )

    tasa_exito_rango = (
        df_con_rangos.groupBy("ia_rango")
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("ai_usage_hours").alias("avg_ia_hours"),
            avg("hours_coding").alias("avg_coding"),
            avg("cognitive_load").alias("avg_cognitive_load"),
            avg("commits").alias("avg_commits"),
            count("*").alias("total_registros"),
        )
        .orderBy("tasa_exito", ascending=False)
    )

    tasa_exito_rango.show()

    # 2.5 Análisis estadístico por grupo de éxito/fracaso
    print("🔍 2.5 Estadísticas por grupo de task_success")
    stats_por_grupo = (
        df_spark.groupBy("task_success")
        .agg(avg("ai_usage_hours").alias("avg_ia"), count("*").alias("total"))
        .orderBy("task_success")
    )
    stats_por_grupo.show()

    # Paso 3 — Convertir a pandas para visualización
    print("\n🔄 Paso 3: Convertir a pandas para visualización")

    df_completo_pd = df_spark.toPandas()
    tasa_exito_rango_pd = tasa_exito_rango.toPandas()
    _ = promedio_ia.toPandas()  # convertido para forzar ejecución del plan Spark

    spark.stop()
    print("✅ Sesión Spark detenida")

    # Paso 4 — Visualizaciones
    print("\n📊 Paso 4: Generando visualizaciones")

    # Crear directorio de resultados si no existe
    os.makedirs("notebooks/results/plan6-uso-ia", exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # Gráfico 1: Boxplot de ai_usage_hours por task_success
    # ─────────────────────────────────────────────────────────────
    print("📈 Gráfico 1: Boxplot de uso de IA por task_success")

    plt.figure(figsize=(12, 8))

    sns.boxplot(
        data=df_completo_pd,
        x="task_success",
        y="ai_usage_hours",
        hue="task_success",
        palette=["#e74c3c", "#2ecc71"],
        legend=False,
    )

    plt.title(
        "Hipótesis IA: Distribución de Horas de Uso de IA por Resultado de Tarea",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Éxito de Tarea  (0 = Fracaso,  1 = Éxito)", fontsize=12)
    plt.ylabel("Horas de Uso de IA", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Anotar medianas sobre cada caja
    for i, success in enumerate([0, 1]):
        subset = df_completo_pd[df_completo_pd["task_success"] == success]
        median = subset["ai_usage_hours"].median()
        mean = subset["ai_usage_hours"].mean()
        plt.text(
            i,
            median + 0.08,
            f"Mediana: {median:.2f}h\nMedia: {mean:.2f}h",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Líneas de referencia de rangos (bajo/medio/alto)
    plt.axhline(
        y=2, color="orange", linestyle="--", alpha=0.7, label="Límite Bajo (2h)"
    )
    plt.axhline(
        y=4, color="royalblue", linestyle="--", alpha=0.7, label="Límite Medio (4h)"
    )
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        "notebooks/results/plan6-uso-ia/plan6_uso_ia_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # ─────────────────────────────────────────────────────────────
    # Gráfico 2: Histograma de ai_usage_hours coloreado por task_success
    # ─────────────────────────────────────────────────────────────
    print("📈 Gráfico 2: Histograma de horas de uso de IA")

    plt.figure(figsize=(14, 8))

    # Separar grupos para histograma superpuesto
    grupo_fracaso = df_completo_pd[df_completo_pd["task_success"] == 0][
        "ai_usage_hours"
    ]
    grupo_exito = df_completo_pd[df_completo_pd["task_success"] == 1]["ai_usage_hours"]

    plt.hist(
        grupo_fracaso,
        bins=20,
        alpha=0.6,
        color="#e74c3c",
        label="Fracaso (task_success=0)",
        edgecolor="white",
    )
    plt.hist(
        grupo_exito,
        bins=20,
        alpha=0.6,
        color="#2ecc71",
        label="Éxito (task_success=1)",
        edgecolor="white",
    )

    plt.title(
        "Hipótesis IA: Distribución de Horas de Uso de IA por Resultado de Tarea",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Horas de Uso de IA", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Líneas de referencia de los rangos
    plt.axvline(
        x=2,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.85,
        label="Límite Bajo/Medio (2h)",
    )
    plt.axvline(
        x=4,
        color="royalblue",
        linestyle="--",
        linewidth=2,
        alpha=0.85,
        label="Límite Medio/Alto (4h)",
    )

    # Etiquetas de rangos sobre las líneas
    y_max = plt.gca().get_ylim()[1]
    plt.text(
        0.9,
        y_max * 0.93,
        "BAJO\n(< 2h)",
        ha="center",
        fontsize=10,
        color="darkorange",
        fontweight="bold",
    )
    plt.text(
        3.0,
        y_max * 0.93,
        "MEDIO\n(2–4h)",
        ha="center",
        fontsize=10,
        color="steelblue",
        fontweight="bold",
    )
    plt.text(
        5.5,
        y_max * 0.93,
        "ALTO\n(> 4h)",
        ha="center",
        fontsize=10,
        color="darkgreen",
        fontweight="bold",
    )

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(
        "notebooks/results/plan6-uso-ia/plan6_uso_ia_histograma.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # ─────────────────────────────────────────────────────────────
    # Gráfico 3: Barplot de tasa de éxito por rango de IA
    # ─────────────────────────────────────────────────────────────
    print("📈 Gráfico 3: Tasa de éxito por rango de uso de IA")

    # Ordenar descendente por tasa de éxito
    tasa_exito_rango_pd_sorted = tasa_exito_rango_pd.sort_values(
        "tasa_exito", ascending=False
    ).reset_index(drop=True)

    # Paleta de colores según rango
    colores_rango = {"bajo": "#e74c3c", "medio": "#f39c12", "alto": "#2ecc71"}
    palette = [
        colores_rango.get(r, "#95a5a6") for r in tasa_exito_rango_pd_sorted["ia_rango"]
    ]

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=tasa_exito_rango_pd_sorted,
        x="ia_rango",
        y="tasa_exito",
        palette=palette,
        order=tasa_exito_rango_pd_sorted["ia_rango"].tolist(),
    )

    plt.title(
        "Hipótesis IA: Tasa de Éxito según Nivel de Uso de Herramientas de IA",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Nivel de Uso de IA", fontsize=12)
    plt.ylabel("Tasa de Éxito", fontsize=12)
    plt.ylim(0, 1.12)
    plt.grid(True, alpha=0.3, axis="y")

    # Etiquetas con porcentaje y cantidad de registros sobre cada barra
    for i, row in tasa_exito_rango_pd_sorted.iterrows():
        ax.text(
            i,
            row["tasa_exito"] + 0.025,
            f"{row['tasa_exito']:.1%}\n({int(row['total_registros'])} devs)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Etiquetas descriptivas en eje X con equivalencias prácticas
    rango_labels = {
        "bajo": "Bajo\n(< 2h/día)",
        "medio": "Medio\n(2–4h/día)",
        "alto": "Alto\n(> 4h/día)",
    }
    new_labels = [
        rango_labels.get(label.get_text(), label.get_text())
        for label in ax.get_xticklabels()
    ]
    ax.set_xticks(range(len(tasa_exito_rango_pd_sorted)))
    ax.set_xticklabels(new_labels, fontsize=11)

    plt.tight_layout()
    plt.savefig(
        "notebooks/results/plan6-uso-ia/plan6_uso_ia_tasa_exito.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # ─────────────────────────────────────────────────────────────
    # Gráfico 4: Scatter de ai_usage_hours vs hours_coding coloreado por task_success
    # ─────────────────────────────────────────────────────────────
    print("📈 Gráfico 4: Scatter ai_usage_hours vs hours_coding")

    plt.figure(figsize=(14, 9))

    # Colorear por task_success: 0=rojo, 1=verde
    colores_scatter = df_completo_pd["task_success"].map({0: "#e74c3c", 1: "#2ecc71"})

    plt.scatter(
        df_completo_pd["ai_usage_hours"],
        df_completo_pd["hours_coding"],
        c=colores_scatter,
        alpha=0.55,
        s=45,
        edgecolors="white",
        linewidths=0.4,
    )

    plt.title(
        "Hipótesis IA: Horas de Uso de IA vs Horas de Código — Coloreado por Éxito de Tarea",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Horas de Uso de Herramientas de IA", fontsize=12)
    plt.ylabel("Horas de Codificación", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Líneas de referencia de rangos de IA
    plt.axvline(
        x=2,
        color="orange",
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label="Límite IA Bajo/Medio (2h)",
    )
    plt.axvline(
        x=4,
        color="royalblue",
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label="Límite IA Medio/Alto (4h)",
    )

    # Leyenda manual para colores de éxito/fracaso
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=10,
            label="Fracaso (task_success=0)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ecc71",
            markersize=10,
            label="Éxito (task_success=1)",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Límite Bajo/Medio (2h)",
        ),
        Line2D(
            [0],
            [0],
            color="royalblue",
            linestyle="--",
            linewidth=2,
            label="Límite Medio/Alto (4h)",
        ),
    ]
    plt.legend(handles=legend_elements, fontsize=10, loc="upper right")

    # Anotación de zona óptima
    plt.axvspan(2, 4, alpha=0.07, color="steelblue", label="Zona Media")
    plt.text(
        2.9,
        df_completo_pd["hours_coding"].max() * 0.97,
        "Zona Media\n(2–4h IA)",
        ha="center",
        fontsize=10,
        color="steelblue",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
    )

    plt.tight_layout()
    plt.savefig(
        "notebooks/results/plan6-uso-ia/plan6_uso_ia_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Paso 5 — Guardar estadísticas en archivo .txt
    print("\n📋 Paso 5: Guardando estadísticas")

    # Precalcular métricas necesarias para el archivo
    grupo_exito_pd = df_completo_pd[df_completo_pd["task_success"] == 1]
    grupo_fracaso_pd = df_completo_pd[df_completo_pd["task_success"] == 0]

    diff_ia = (
        grupo_exito_pd["ai_usage_hours"].mean()
        - grupo_fracaso_pd["ai_usage_hours"].mean()
    )

    # Determinar veredicto según lógica definida
    if correlacion > 0.5:
        veredicto = "CONFIRMADA"
        veredicto_simbolo = "✅"
        veredicto_desc = "Evidencia fuerte que apoya la hipótesis original"
    elif correlacion >= 0.3:
        veredicto = "CONFIRMADA PARCIALMENTE"
        veredicto_simbolo = "🔶"
        veredicto_desc = "Evidencia moderada que apoya la hipótesis original"
    elif correlacion >= 0.1:
        veredicto = "DÉBIL"
        veredicto_simbolo = "⚠️"
        veredicto_desc = "Correlación baja, relación existe pero es débil"
    elif correlacion < -0.1:
        veredicto = "REFUTADA"
        veredicto_simbolo = "❌"
        veredicto_desc = "Evidencia contraria a la hipótesis original"
    else:
        veredicto = "NEUTRA / REFUTADA"
        veredicto_simbolo = "🔄"
        veredicto_desc = "Evidencia insuficiente para confirmar la hipótesis"

    # Mejor rango por tasa de éxito
    mejor_rango_row = tasa_exito_rango_pd.loc[
        tasa_exito_rango_pd["tasa_exito"].idxmax()
    ]
    mejor_rango = mejor_rango_row["ia_rango"]
    mejor_tasa = mejor_rango_row["tasa_exito"]

    with open(
        "notebooks/results/plan6-uso-ia/plan6_uso_ia_estadisticas.txt",
        "w",
        encoding="utf-8",
    ) as f:
        # Sección 1
        f.write("=== PLAN 6: ANÁLISIS DE HIPÓTESIS USO DE IA ===\n")
        f.write(
            "Hipótesis: 'Mayor uso de herramientas de IA está asociado con mayor task_success'\n"
        )
        f.write(f"Resultado preliminar: {veredicto_simbolo} {veredicto}\n\n")

        # Sección 2
        f.write("=== METODOLOGÍA ===\n")
        f.write("• Dataset: 500 registros de productividad de desarrolladores\n")
        f.write(
            "• Variable independiente: ai_usage_hours (horas diarias usando herramientas de IA)\n"
        )
        f.write("• Variable dependiente: task_success (0=fracaso, 1=éxito)\n")
        f.write("• Análisis: Correlación de Pearson + segmentación por rangos\n")
        f.write(
            "• Herramientas: PySpark para procesamiento distribuido, pandas/matplotlib/seaborn para visualización\n"
        )
        f.write(
            "• Visualizaciones: boxplot, histograma, barplot de tasa de éxito, scatter\n\n"
        )

        # Sección 3
        f.write("=== DEFINICIÓN DE RANGOS DE USO DE IA ===\n")
        f.write(
            "• Rango BAJO:  < 2 horas/día  (uso esporádico o nulo de herramientas IA)\n"
        )
        f.write(
            "• Rango MEDIO: 2–4 horas/día  (uso moderado, integrado en el flujo de trabajo)\n"
        )
        f.write(
            "• Rango ALTO:  > 4 horas/día  (uso intensivo, IA como parte central del trabajo)\n"
        )
        f.write(
            "• Referencia: 2h representa ~25% de una jornada de 8h; 4h representa ~50%\n\n"
        )

        # Sección 4
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        f.write(
            f"Correlación de Pearson (ai_usage_hours vs task_success): {correlacion:.4f}\n"
        )
        f.write(
            f"Correlación (ai_usage_hours vs commits):                  {correlacion_ia_commits:.4f}\n"
        )
        f.write(
            f"Correlación (ai_usage_hours vs bugs_reported):            {correlacion_ia_bugs:.4f}\n"
        )
        f.write(
            f"Correlación (ai_usage_hours vs cognitive_load):           {correlacion_ia_cognitiva:.4f}\n"
        )
        f.write(
            f"Correlación (ai_usage_hours vs hours_coding):             {correlacion_ia_horas:.4f}\n\n"
        )

        f.write("Escala de referencia para correlación de Pearson:\n")
        f.write("  0.0–0.1  → Correlación nula o insignificante\n")
        f.write("  0.1–0.3  → Correlación débil\n")
        f.write("  0.3–0.5  → Correlación moderada\n")
        f.write("  0.5–0.7  → Correlación fuerte\n")
        f.write("  0.7–1.0  → Correlación muy fuerte\n\n")

        f.write("Estadísticas de ai_usage_hours por grupo:\n\n")
        for success in [0, 1]:
            subset = df_completo_pd[df_completo_pd["task_success"] == success]
            grupo = "FRACASO" if success == 0 else "ÉXITO"
            f.write(f"  Grupo {grupo} (task_success = {success}):\n")
            f.write(f"    • Total registros:     {len(subset)}\n")
            f.write(
                f"    • Promedio IA:         {subset['ai_usage_hours'].mean():.3f} h\n"
            )
            f.write(
                f"    • Mediana IA:          {subset['ai_usage_hours'].median():.3f} h\n"
            )
            f.write(
                f"    • Desviación estándar: {subset['ai_usage_hours'].std():.3f} h\n"
            )
            f.write(
                f"    • Mínimo:              {subset['ai_usage_hours'].min():.2f} h\n"
            )
            f.write(
                f"    • Máximo:              {subset['ai_usage_hours'].max():.2f} h\n"
            )
            f.write(
                f"    • Percentil 25:        {subset['ai_usage_hours'].quantile(0.25):.2f} h\n"
            )
            f.write(
                f"    • Percentil 75:        {subset['ai_usage_hours'].quantile(0.75):.2f} h\n\n"
            )

        f.write("Diferencia clave entre grupos:\n")
        f.write(
            f"  • Developers con éxito usan {diff_ia:.3f} h más de IA en promedio\n"
        )
        if grupo_fracaso_pd["ai_usage_hours"].mean() > 0:
            pct_diff = (diff_ia / grupo_fracaso_pd["ai_usage_hours"].mean()) * 100
            f.write(
                f"  • Esto representa un {pct_diff:.1f}% más de uso de IA vs el grupo de fracaso\n\n"
            )
        else:
            f.write(
                "  • (No se puede calcular el porcentaje: media del grupo fracaso es 0)\n\n"
            )

        # Sección 5
        f.write("=== ANÁLISIS POR RANGOS ===\n")
        f.write("Tasa de éxito y características por nivel de uso de IA:\n\n")

        for _, row in tasa_exito_rango_pd.sort_values(
            "tasa_exito", ascending=False
        ).iterrows():
            rango = row["ia_rango"]
            f.write(f"  Rango {rango.upper()}:\n")
            f.write(
                f"    • Tasa de éxito:            {row['tasa_exito']:.1%}  ({row['tasa_exito']:.4f})\n"
            )
            f.write(f"    • Horas de IA promedio:     {row['avg_ia_hours']:.2f} h\n")
            f.write(f"    • Horas de código promedio: {row['avg_coding']:.2f} h\n")
            f.write(
                f"    • Carga cognitiva promedio: {row['avg_cognitive_load']:.2f}\n"
            )
            f.write(f"    • Commits promedio:         {row['avg_commits']:.2f}\n")
            f.write(f"    • Total registros:          {int(row['total_registros'])}\n")
            f.write(
                f"    • Proporción del dataset:   {(row['total_registros'] / len(df_completo_pd) * 100):.1f}%\n"
            )

            if rango == "bajo":
                interpretacion = "Uso esporádico de IA; puede indicar flujo tradicional sin asistencia"
            elif rango == "medio":
                interpretacion = (
                    "Uso moderado equilibrado; IA como complemento al trabajo humano"
                )
            else:
                interpretacion = (
                    "Uso intensivo de IA; workflow altamente asistido por herramientas"
                )

            f.write(f"    • Interpretación:           {interpretacion}\n\n")

        # Sección 6
        f.write("=== INSIGHTS CLAVE ===\n")
        f.write("1. RELACIÓN DIRECTA IA → ÉXITO:\n")
        if correlacion > 0.3:
            f.write(
                f"   • Correlación positiva moderada-fuerte ({correlacion:.3f}) confirma asociación clara\n"
            )
            f.write(
                "   • Mayor uso de IA se traduce en mayor probabilidad de completar tareas con éxito\n"
            )
        elif correlacion > 0.1:
            f.write(
                f"   • Correlación positiva débil ({correlacion:.3f}): hay asociación pero es limitada\n"
            )
            f.write(
                "   • Otros factores probablemente influyen más en el éxito de tareas\n"
            )
        elif correlacion < -0.1:
            f.write(
                f"   • Correlación negativa inesperada ({correlacion:.3f}): más IA → menos éxito\n"
            )
            f.write(
                "   • Posible sobredependencia de IA que reduce la comprensión del problema\n"
            )
        else:
            f.write(
                f"   • Correlación prácticamente nula ({correlacion:.3f}): sin asociación lineal clara\n"
            )
            f.write(
                "   • El uso de IA no determina el éxito de manera estadísticamente relevante\n"
            )

        f.write("\n2. RANGO ÓPTIMO DE USO DE IA:\n")
        f.write(
            f"   • El rango '{mejor_rango}' alcanza la mayor tasa de éxito: {mejor_tasa:.1%}\n"
        )

        rango_bajo = tasa_exito_rango_pd[tasa_exito_rango_pd["ia_rango"] == "bajo"]
        rango_medio = tasa_exito_rango_pd[tasa_exito_rango_pd["ia_rango"] == "medio"]
        rango_alto = tasa_exito_rango_pd[tasa_exito_rango_pd["ia_rango"] == "alto"]

        if not rango_bajo.empty and not rango_alto.empty:
            tasa_baja = rango_bajo["tasa_exito"].iloc[0]
            tasa_alta = rango_alto["tasa_exito"].iloc[0]
            f.write(f"   • Rango bajo  → {tasa_baja:.1%} de éxito\n")
            f.write(f"   • Rango alto  → {tasa_alta:.1%} de éxito\n")
            diferencia_extremos = tasa_alta - tasa_baja
            f.write(f"   • Diferencia entre extremos: {diferencia_extremos:+.1%}\n")

        f.write("\n3. CORRELACIONES SECUNDARIAS:\n")
        f.write(
            f"   • IA vs commits:        {correlacion_ia_commits:.3f}  — ¿Genera más commits el uso de IA?\n"
        )
        f.write(
            f"   • IA vs bugs_reported:  {correlacion_ia_bugs:.3f}  — ¿Afecta la calidad del código?\n"
        )
        f.write(
            f"   • IA vs cognitive_load: {correlacion_ia_cognitiva:.3f}  — ¿Reduce la carga mental?\n"
        )
        f.write(
            f"   • IA vs hours_coding:   {correlacion_ia_horas:.3f}  — ¿Complementa o reemplaza el código manual?\n"
        )

        f.write("\n4. DISTRIBUCIÓN DEL USO:\n")
        f.write(
            f"   • Total registros en rango BAJO:  {int(rango_bajo['total_registros'].iloc[0]) if not rango_bajo.empty else 0}\n"
        )
        f.write(
            f"   • Total registros en rango MEDIO: {int(rango_medio['total_registros'].iloc[0]) if not rango_medio.empty else 0}\n"
        )
        f.write(
            f"   • Total registros en rango ALTO:  {int(rango_alto['total_registros'].iloc[0]) if not rango_alto.empty else 0}\n"
        )
        f.write(
            f"   • Media global de ai_usage_hours: {df_completo_pd['ai_usage_hours'].mean():.3f} h\n"
        )
        f.write(
            f"   • Mediana global de ai_usage_hours: {df_completo_pd['ai_usage_hours'].median():.3f} h\n\n"
        )

        # Sección 7
        f.write("=== VEREDICTO FINAL ===\n")
        f.write(f"{veredicto_simbolo} HIPÓTESIS {veredicto}\n")
        f.write(f"{veredicto_desc}\n\n")
        f.write("Métricas de decisión:\n")
        f.write(f"  • Correlación observada:   {correlacion:.4f}\n")
        f.write("  • Umbral CONFIRMADA:        > 0.50\n")
        f.write("  • Umbral CONFIRMADA PARCIALMENTE: 0.30–0.50\n")
        f.write("  • Umbral DÉBIL:             0.10–0.30\n")
        f.write("  • Umbral REFUTADA:          < 0.10 o negativo\n")
        f.write(f"  • Diferencia en medias:     {diff_ia:+.3f} h (éxito vs fracaso)\n")
        f.write(f"  • Rango con mayor éxito:    '{mejor_rango}' ({mejor_tasa:.1%})\n\n")

        # Sección 8
        f.write("=== RECOMENDACIONES PRÁCTICAS ===\n")
        if mejor_rango == "alto":
            f.write(
                "• Priorizar el uso intensivo de IA (> 4h/día) para maximizar el éxito de tareas\n"
            )
            f.write(
                "• Integrar herramientas de IA como GitHub Copilot, ChatGPT o similares en el flujo diario\n"
            )
            f.write(
                "• Evaluar si el equipo tiene capacitación adecuada para aprovechar herramientas de IA\n"
            )
        elif mejor_rango == "medio":
            f.write(
                "• Mantener uso moderado de IA (2–4h/día) como punto de equilibrio óptimo\n"
            )
            f.write(
                "• Evitar tanto el sub-uso (< 2h) como el sobre-uso (> 4h) de herramientas de IA\n"
            )
            f.write(
                "• Complementar la IA con pensamiento crítico propio para no perder comprensión profunda\n"
            )
        else:
            f.write(
                "• El uso básico de IA (< 2h/día) parece suficiente en este dataset\n"
            )
            f.write("• Evaluar si el tipo de tareas requiere mayor asistencia de IA\n")
            f.write("• Fomentar adopción progresiva sin forzar uso intensivo\n")

        f.write(
            "\n• Medir el impacto de las herramientas IA antes y después de su adopción\n"
        )
        f.write(
            "• Combinar métricas cuantitativas (horas) con cualitativas (tipo de uso)\n"
        )
        f.write(
            "• Considerar talleres sobre uso efectivo de IA para developers con rango 'bajo'\n\n"
        )

        # Sección 9
        f.write("=== LIMITACIONES DEL ANÁLISIS ===\n")
        f.write(
            "• Las horas de uso de IA son auto-reportadas; pueden no reflejar la realidad exacta\n"
        )
        f.write(
            "• Correlación no implica causalidad: el éxito puede deberse a otros factores\n"
        )
        f.write(
            "• Dataset limitado a 500 registros; puede no representar toda la población de developers\n"
        )
        f.write(
            "• No se controla por tipo de herramienta IA (autocomplete, chat, generación de código)\n"
        )
        f.write("• No se mide la calidad del uso de IA, solo la cantidad de horas\n")
        f.write(
            "• Factores como experiencia del developer, tipo de proyecto o presión de sprint no están incluidos\n"
        )
        f.write(
            "• La definición de 'task_success' puede variar según el contexto del proyecto\n"
        )
        f.write(
            "• Posible sesgo de selección: developers más productivos pueden reportar más uso de IA\n"
        )

    print("✅ Archivo de estadísticas guardado")

    # Paso 6 — Resumen en consola
    print("\n" + "=" * 60)
    print("=== RESUMEN PLAN 6: HIPÓTESIS USO DE IA ===")
    print("=" * 60)
    print(f"Correlación ai_usage_hours vs task_success: {correlacion:.4f}")
    print(f"Correlación ai_usage_hours vs commits:       {correlacion_ia_commits:.4f}")
    print(f"Correlación ai_usage_hours vs bugs_reported: {correlacion_ia_bugs:.4f}")
    print(
        f"Correlación ai_usage_hours vs cognitive_load:{correlacion_ia_cognitiva:.4f}"
    )

    print("\n=== ESTADÍSTICAS POR GRUPO ===")
    for success in [0, 1]:
        subset = df_completo_pd[df_completo_pd["task_success"] == success]
        grupo = "FRACASO" if success == 0 else "ÉXITO"
        print(f"\nTask Success = {success} ({grupo}):")
        print(f"  Promedio ai_usage_hours: {subset['ai_usage_hours'].mean():.3f} h")
        print(f"  Mediana  ai_usage_hours: {subset['ai_usage_hours'].median():.3f} h")
        print(f"  Total registros:         {len(subset)}")

    print("\n=== ANÁLISIS POR RANGO DE IA ===")
    for _, row in tasa_exito_rango_pd.sort_values(
        "tasa_exito", ascending=False
    ).iterrows():
        print(
            f"Rango {row['ia_rango']:>5}: {row['tasa_exito']:.1%} de éxito  "
            f"({int(row['total_registros'])} registros)"
        )

    print(f"\n{veredicto_simbolo} VEREDICTO: HIPÓTESIS {veredicto}")
    print(f"📊 Correlación observada: {correlacion:.4f}")
    print(f"🏆 Mejor rango: '{mejor_rango}' con {mejor_tasa:.1%} de éxito")

    print("\n✅ Plan 6 completado exitosamente")
    print("📁 Archivos guardados en notebooks/results/plan6-uso-ia/")
    print("   - plan6_uso_ia_boxplot.png")
    print("   - plan6_uso_ia_histograma.png")
    print("   - plan6_uso_ia_tasa_exito.png")
    print("   - plan6_uso_ia_scatter.png")
    print("   - plan6_uso_ia_estadisticas.txt")


if __name__ == "__main__":
    main()
