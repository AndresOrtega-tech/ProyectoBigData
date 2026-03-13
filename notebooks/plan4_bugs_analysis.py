#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan 4: Hipótesis Bugs Reportados - Calidad vs Éxito
Analizar si reportar más bugs indica menor probabilidad de éxito y explorar la relación entre cantidad y calidad del código.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, sum as spark_sum, expr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("🧪 Iniciando Plan 4: Análisis de Hipótesis Bugs Reportados")
    
    # Paso 1 — Configuración inicial
    print("\n📊 Paso 1: Configuración inicial")
    
    spark = SparkSession.builder \
        .appName("Hipotesis_Bugs_Reportados") \
        .getOrCreate()
    
    # Cargar dataset con manejo robusto de rutas
    dataset_path = "../data/ai_dev_productivity.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "/Users/andrestamez5/Personal/BigDataProject/data/ai_dev_productivity.csv"
    
    print(f"📁 Usando dataset: {dataset_path}")
    df_spark = spark.read.csv(dataset_path, header=True, inferSchema=True)
    
    print(f"✅ Dataset cargado: {df_spark.count()} registros")
    
    # Paso 2 — Análisis en PySpark
    print("\n📈 Paso 2: Análisis en PySpark")
    
    # 2.1 Estadísticas básicas de bugs
    print("🔍 2.1 Estadísticas básicas de bugs")
    distribucion_bugs = df_spark.groupBy("bugs_reported") \
        .agg(
            count("*").alias("frecuencia"),
            avg("task_success").alias("tasa_exito"),
            avg("commits").alias("avg_commits"),
            avg("hours_coding").alias("avg_hours")
        ) \
        .orderBy("bugs_reported")
    
    distribucion_bugs.show()
    
    # 2.2 Calcular correlaciones
    print("🔍 2.2 Calcular correlaciones")
    correlacion_bugs = df_spark.stat.corr("bugs_reported", "task_success")
    print(f"Correlación bugs_reported vs task_success: {correlacion_bugs:.3f}")
    
    correlacion_commits_bugs = df_spark.stat.corr("bugs_reported", "commits")
    print(f"Correlación bugs_reported vs commits: {correlacion_commits_bugs:.3f}")
    
    correlacion_horas_bugs = df_spark.stat.corr("bugs_reported", "hours_coding")
    print(f"Correlación bugs_reported vs hours_coding: {correlacion_horas_bugs:.3f}")
    
    # 2.3 Análisis de la mediana y distribución
    print("🔍 2.3 Análisis de distribución de bugs")
    stats_bugs = df_spark.select(
        expr("percentile_approx(bugs_reported, 0.5)").alias("mediana"),
        expr("percentile_approx(bugs_reported, 0.25)").alias("q1"),
        expr("percentile_approx(bugs_reported, 0.75)").alias("q3"),
        expr("avg(bugs_reported)").alias("promedio"),
        count("*").alias("total_registros")
    ).collect()[0]
    
    print("=== ESTADÍSTICAS DE BUGS REPORTADOS ===")
    print(f"Mediana: {stats_bugs['mediana']}")
    print(f"Promedio: {stats_bugs['promedio']:.2f}")
    print(f"Q1 (25%): {stats_bugs['q1']}")
    print(f"Q3 (75%): {stats_bugs['q3']}")
    print(f"Total registros: {stats_bugs['total_registros']}")
    
    cero_bugs = df_spark.filter(col("bugs_reported") == 0).count()
    porcentaje_cero = (cero_bugs / stats_bugs['total_registros']) * 100
    print(f"Registros con 0 bugs: {cero_bugs} ({porcentaje_cero:.1f}%)")
    
    # 2.4 Análisis por categorías de bugs
    print("🔍 2.4 Análisis por categorías de bugs")
    df_con_categorias = df_spark.withColumn(
        "bugs_categoria",
        when(col("bugs_reported") == 0, "cero_bugs")
        .when(col("bugs_reported") == 1, "un_bug")
        .when(col("bugs_reported") == 2, "dos_bugs")
        .otherwise("tres_o_mas")
    )
    
    categoria_stats = df_con_categorias.groupBy("bugs_categoria") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("commits").alias("avg_commits"),
            avg("hours_coding").alias("avg_hours"),
            avg("coffee_intake_mg").alias("avg_coffee"),
            count("*").alias("total_registros")
        ) \
        .orderBy("tasa_exito", ascending=False)
    
    categoria_stats.show()
    
    # 2.5 Análisis de productividad con bugs
    print("🔍 2.5 Análisis de productividad con bugs")
    df_con_productividad = df_spark.filter(col("bugs_reported") > 0) \
        .withColumn("commits_por_bug", col("commits") / col("bugs_reported"))
    
    productividad_bugs = df_con_productividad.agg(
        avg("commits_por_bug").alias("avg_commits_per_bug"),
        avg("task_success").alias("tasa_exito_con_bugs"),
        count("*").alias("registros_con_bugs")
    ).collect()[0]
    
    print("=== PRODUCTIVIDAD CON BUGS ===")
    print(f"Commits por bug (promedio): {productividad_bugs['avg_commits_per_bug']:.2f}")
    print(f"Tasa éxito con bugs: {productividad_bugs['tasa_exito_con_bugs']:.1%}")
    print(f"Registros con bugs: {productividad_bugs['registros_con_bugs']}")
    
    # Paso 3 — Convertir a pandas para visualización
    print("\n🔄 Paso 3: Convertir a pandas para visualización")
    
    distribucion_bugs_pd = distribucion_bugs.toPandas()
    categoria_stats_pd = categoria_stats.toPandas()
    df_completo_pd = df_spark.toPandas()
    
    spark.stop()
    print("✅ Sesión Spark detenida")
    
    # Paso 4 — Visualizaciones
    print("\n📊 Paso 4: Generando visualizaciones")
    
    # Crear directorio de resultados
    os.makedirs("notebooks/results/plan4-bugs-reportados", exist_ok=True)
    
    # Gráfico 1: Barplot de conteo de bugs por task_success
    print("📈 Gráfico 1: Distribución de bugs reportados por éxito")
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df_completo_pd, x="bugs_reported", hue="task_success")
    plt.title("Hipótesis Bugs Reportados: Distribución por Éxito de Tarea", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Número de Bugs Reportados", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.legend(title="Éxito", labels=["Fracaso", "Éxito"])
    plt.grid(True, alpha=0.3)
    
    # Agregar porcentajes en las barras
    ax = plt.gca()
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan4-bugs-reportados/plan4_bugs_boxplot.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 2: Barplot de tasa de éxito por número exacto de bugs
    print("📈 Gráfico 2: Tasa de éxito por número de bugs")
    plt.figure(figsize=(14, 8))
    
    ax = sns.barplot(data=distribucion_bugs_pd, x="bugs_reported", y="tasa_exito")
    plt.title("Hipótesis Bugs Reportados: Tasa de Éxito por Número de Bugs", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Número de Bugs Reportados", fontsize=12)
    plt.ylabel("Tasa de Éxito", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Agregar etiquetas de valores y frecuencias
    for i, row in distribucion_bugs_pd.iterrows():
        ax.text(i, row.tasa_exito + 0.02, f"{row.tasa_exito:.1%}\n(n={row['frecuencia']})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Agregar línea de referencia para tasa de éxito general
    tasa_general = df_completo_pd['task_success'].mean()
    plt.axhline(y=tasa_general, color='red', linestyle='--', alpha=0.7, 
                label=f'Tasa general: {tasa_general:.1%}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan4-bugs-reportados/plan4_bugs_tasa_exito.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 3: Scatter plot de commits vs bugs_reported coloreado por task_success
    print("📈 Gráfico 3: Relación commits vs bugs")
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df_completo_pd, x="bugs_reported", y="commits", 
                    hue="task_success", size="hours_coding", alpha=0.7, sizes=(20, 200))
    plt.title("Hipótesis Bugs Reportados: Relación Cantidad vs Calidad", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Bugs Reportados", fontsize=12)
    plt.ylabel("Commits Realizados", fontsize=12)
    plt.legend(title="Éxito", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Agregar línea de tendencia
    z = np.polyfit(df_completo_pd['bugs_reported'], df_completo_pd['commits'], 1)
    p = np.poly1d(z)
    plt.plot(df_completo_pd['bugs_reported'], p(df_completo_pd['bugs_reported']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan4-bugs-reportados/plan4_bugs_scatter.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 4: Comparación de métricas por categoría de bugs
    print("📈 Gráfico 4: Métricas por categoría de bugs")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hipótesis Bugs Reportados: Análisis por Categoría", fontsize=16, fontweight='bold')
    
    # Tasa de éxito por categoría
    sns.barplot(data=categoria_stats_pd, x="bugs_categoria", y="tasa_exito", ax=axes[0,0])
    axes[0,0].set_title("Tasa de Éxito por Categoría de Bugs")
    axes[0,0].set_ylabel("Tasa de Éxito")
    axes[0,0].set_ylim(0, 1)
    
    # Agregar etiquetas de porcentaje
    for i, row in categoria_stats_pd.iterrows():
        axes[0,0].text(i, row.tasa_exito + 0.02, f"{row.tasa_exito:.1%}", 
                        ha='center', va='bottom', fontweight='bold')
    
    # Commits promedio por categoría
    sns.barplot(data=categoria_stats_pd, x="bugs_categoria", y="avg_commits", ax=axes[0,1])
    axes[0,1].set_title("Commits Promedio por Categoría")
    axes[0,1].set_ylabel("Commits Promedio")
    
    # Horas promedio por categoría
    sns.barplot(data=categoria_stats_pd, x="bugs_categoria", y="avg_hours", ax=axes[1,0])
    axes[1,0].set_title("Horas Promedio por Categoría")
    axes[1,0].set_ylabel("Horas Promedio")
    
    # Cafeína promedio por categoría
    sns.barplot(data=categoria_stats_pd, x="bugs_categoria", y="avg_coffee", ax=axes[1,1])
    axes[1,1].set_title("Cafeína Promedio por Categoría")
    axes[1,1].set_ylabel("Cafeína (mg)")
    
    # Agregar descripción de categorías
    categoria_labels = {
        'cero_bugs': '0 bugs\n(Código limpio)',
        'un_bug': '1 bug\n(Error menor)',
        'dos_bugs': '2 bugs\n(Problemas moderados)',
        'tres_o_mas': '3+ bugs\n(Múltiples problemas)'
    }
    
    for ax in axes.flat:
        current_labels = [label.get_text() for label in ax.get_xticklabels()]
        new_labels = [categoria_labels.get(label, label) for label in current_labels]
        ax.set_xticklabels(new_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan4-bugs-reportados/plan4_bugs_categoria.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Paso 5 — Análisis de resultados
    print("\n📋 Paso 5: Análisis de resultados")
    
    # Guardar estadísticas en archivo
    with open("notebooks/results/plan4-bugs-reportados/plan4_bugs_estadisticas.txt", "w", encoding='utf-8') as f:
        f.write("=== PLAN 4: ANÁLISIS DE HIPÓTESIS BUGS REPORTADOS ===\n")
        f.write("Hipótesis: 'Más bugs reportados indica menor probabilidad de éxito'\n")
        f.write("Correlación esperada: -0.18 (muy débil negativa)\n\n")
        
        f.write("=== METODOLOGÍA ===\n")
        f.write("• Dataset: 500 registros de productividad de desarrolladores\n")
        f.write("• Variable independiente: bugs_reported (número de bugs detectados)\n")
        f.write("• Variable dependiente: task_success (0=fallo, 1=éxito)\n")
        f.write("• Análisis: Correlación de Pearson + distribución + categorías\n")
        f.write("• Herramientas: PySpark para procesamiento, pandas/matplotlib para visualización\n\n")
        
        f.write("=== DEFINICIÓN DE CATEGORÍAS DE BUGS ===\n")
        f.write("• Cero bugs: 0 bugs detectados (código limpio, sin problemas evidentes)\n")
        f.write("• Un bug: 1 bug detectado (error menor, fácil de corregir)\n")
        f.write("• Dos bugs: 2 bugs detectados (problemas moderados, requiere atención)\n")
        f.write("• Tres o más: 3+ bugs detectados (múltiples problemas, posible refactorización)\n")
        f.write("• Referencia: Número absoluto de bugs reportados durante la sesión\n\n")
        
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        f.write(f"Correlación de Pearson: {correlacion_bugs:.3f}\n")
        f.write("Interpretación de correlación:\n")
        f.write("• -0.1 a -0.3: Correlación negativa muy débil\n")
        f.write("• -0.3 a -0.5: Correlación negativa débil\n")
        f.write("• 0.0 a ±0.1: Sin correlación prácticamente nula\n")
        f.write("• +0.1 a +0.3: Correlación positiva muy débil\n\n")
        
        f.write("Distribución de bugs reportados:\n")
        f.write(f"• Mediana: {stats_bugs['mediana']} bugs\n")
        f.write(f"• Promedio: {stats_bugs['promedio']:.2f} bugs\n")
        f.write(f"• Q1 (25%): {stats_bugs['q1']} bugs\n")
        f.write(f"• Q3 (75%): {stats_bugs['q3']} bugs\n")
        f.write(f"• Registros con 0 bugs: {cero_bugs} ({porcentaje_cero:.1f}%)\n\n")
        
        f.write("Análisis por grupo de éxito:\n\n")
        
        for success in [0, 1]:
            subset = df_completo_pd[df_completo_pd['task_success'] == success]
            grupo = "FRACASO" if success == 0 else "ÉXITO"
            f.write(f"Grupo {grupo} (task_success = {success}):\n")
            f.write(f"  • Promedio bugs: {subset['bugs_reported'].mean():.2f}\n")
            f.write(f"  • Mediana bugs: {subset['bugs_reported'].median():.1f}\n")
            f.write(f"  • Desviación estándar: {subset['bugs_reported'].std():.2f}\n")
            f.write(f"  • Mínimo: {subset['bugs_reported'].min():.0f}\n")
            f.write(f"  • Máximo: {subset['bugs_reported'].max():.0f}\n")
            f.write(f"  • Total registros: {len(subset)}\n\n")
        
        f.write("=== ANÁLISIS POR NÚMERO EXACTO DE BUGS ===\n")
        f.write("Tasa de éxito y características por número de bugs:\n\n")
        
        for _, row in distribucion_bugs_pd.iterrows():
            bugs_num = int(row['bugs_reported'])
            f.write(f"{bugs_num} bugs:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Frecuencia: {row['frecuencia']} registros\n")
            f.write(f"  • Commits promedio: {row['avg_commits']:.1f}\n")
            f.write(f"  • Horas promedio: {row['avg_hours']:.1f} h\n")
            
            # Interpretación
            if bugs_num == 0:
                interpretacion = "Código sin problemas detectados"
            elif bugs_num == 1:
                interpretacion = "Error menor, impacto limitado"
            elif bugs_num == 2:
                interpretacion = "Problemas moderados que afectan rendimiento"
            else:
                interpretacion = "Múltiples problemas, posible refactorización necesaria"
            
            f.write(f"  • Interpretación: {interpretacion}\n\n")
        
        f.write("=== ANÁLISIS POR CATEGORÍAS ===\n")
        f.write("Comportamiento y patrones por categoría de bugs:\n\n")
        
        for _, row in categoria_stats_pd.iterrows():
            categoria = row['bugs_categoria']
            f.write(f"Categoría {categoria.replace('_', ' ').upper()}:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Commits promedio: {row['avg_commits']:.1f}\n")
            f.write(f"  • Horas promedio: {row['avg_hours']:.1f} h\n")
            f.write(f"  • Cafeína promedio: {row['avg_coffee']:.0f} mg\n")
            f.write(f"  • Total registros: {row['total_registros']}\n")
            f.write(f"  • Proporción del dataset: {(row['total_registros']/len(df_completo_pd)*100):.1f}%\n\n")
        
        f.write("=== PRODUCTIVIDAD CON BUGS ===\n")
        f.write(f"Análisis limitado a registros con bugs > 0 ({productividad_bugs['registros_con_bugs']} registros):\n")
        f.write(f"• Commits por bug (promedio): {productividad_bugs['avg_commits_per_bug']:.2f}\n")
        f.write(f"• Tasa de éxito con bugs: {productividad_bugs['tasa_exito_con_bugs']:.1%}\n\n")
        
        f.write("=== INSIGHTS CLAVE ===\n")
        f.write("1. DISTRIBUCIÓN ESPECIAL:\n")
        f.write(f"   • Mediana = {stats_bugs['mediana']} bugs, mayoría sin problemas detectados\n")
        f.write(f"   • {porcentaje_cero:.1f}% de las sesiones no reportan bugs\n")
        if porcentaje_cero > 50:
            f.write("   • Podría indicar tareas simples, falta de detección, o alta calidad\n")
        
        f.write("2. CORRELACIÓN DÉBIL:\n")
        if abs(correlacion_bugs) < 0.1:
            f.write(f"   • Correlación prácticamente nula ({correlacion_bugs:.3f})\n")
            f.write("   • Los bugs no predicen significativamente el éxito\n")
        elif correlacion_bugs < -0.1:
            f.write(f"   • Correlación negativa débil ({correlacion_bugs:.3f})\n")
            f.write("   • Más bugs se asocian ligeramente con menor éxito\n")
        else:
            f.write(f"   • Correlación positiva inesperada ({correlacion_bugs:.3f})\n")
            f.write("   • Más bugs se asocian con mayor éxito (contradictorio)\n")
        
        f.write("3. CALIDAD VS CANTIDAD:\n")
        if abs(correlacion_commits_bugs) > 0.3:
            f.write(f"   • Más commits se asocian con más bugs (r={correlacion_commits_bugs:.3f})\n")
            f.write("   • Mayor complejidad o actividad genera más problemas\n")
        elif correlacion_commits_bugs < -0.3:
            f.write(f"   • Más commits se asocian con menos bugs (r={correlacion_commits_bugs:.3f})\n")
            f.write("   • Mayor esfuerzo puede mejorar calidad\n")
        else:
            f.write(f"   • Sin relación clara entre commits y bugs (r={correlacion_commits_bugs:.3f})\n")
        
        f.write("4. UMPRAL DE TOLERANCIA:\n")
        if not distribucion_bugs_pd.empty:
            max_bugs_exitosos = distribucion_bugs_pd[distribucion_bugs_pd['tasa_exito'] > 0.5]['bugs_reported'].max()
            f.write(f"   • Hasta {max_bugs_exitosos:.0f} bugs aún permiten >50% de éxito\n")
            f.write("   • Sugiere umbral de tolerancia para problemas menores\n")
        
        f.write("\n=== COMPARACIÓN DE IMPORTANCIA ===\n")
        f.write("Impacto relativo en task_success:\n")
        f.write("• Cafeína vs Éxito: ~0.70 (muy fuerte positivo)\n")
        f.write("• Horas vs Éxito: ~0.62 (fuerte positivo)\n")
        f.write("• Carga Cognitiva vs Éxito: ~-0.20 (débil negativo)\n")
        f.write(f"• Bugs vs Éxito: {correlacion_bugs:.3f} (muy débil)\n")
        f.write("\nConclusión: Los bugs tienen el menor impacto directo en el éxito\n")
        
        f.write("\n=== VEREDICTO FINAL ===\n")
        if abs(correlacion_bugs) < 0.1:
            veredicto = "HIPÓTESIS REFUTADA"
            f.write(f"❌ {veredicto}\n")
            f.write("Evidencia insuficiente para apoyar la hipótesis original\n")
        elif correlacion_bugs < -0.1:
            veredicto = "HIPÓTESIS CONFIRMADA (muy débil)"
            f.write(f"✅ {veredicto}\n")
            f.write("Evidencia muy débil que apoya la hipótesis original\n")
        else:
            veredicto = "HIPÓTESIS INVERSA"
            f.write(f"🔄 {veredicto}\n")
            f.write("Evidencia contraria a la hipótesis original\n")
        
        f.write(f"\nMétricas clave:\n")
        f.write(f"• Correlación observada: {correlacion_bugs:.3f} (esperada: -0.18)\n")
        f.write(f"• Diferencia con esperada: {abs(correlacion_bugs + 0.18):.3f}\n")
        f.write(f"• Registros sin bugs: {porcentaje_cero:.1f}%\n")
        
        f.write("\n=== RECOMENDACIONES PRÁCTICAS ===\n")
        if abs(correlacion_bugs) < 0.1:
            f.write("• Enfocarse en otros factores (cafeína, horas) con mayor impacto\n")
            f.write("• Los bugs no son un predictor fuerte del éxito\n")
            f.write("• Mantener prácticas de calidad pero sin obsesionarse con conteo de bugs\n")
        elif correlacion_bugs < -0.1:
            f.write("• Monitorear número de bugs como indicador de riesgo\n")
            f.write("• Implementar code review para reducir bugs críticos\n")
            f.write("• Considerar refactorización cuando bugs > umbral identificado\n")
        else:
            f.write("• Investigar por qué más bugs se asocian con mayor éxito\n")
            f.write("• Podría indicar mejor detección o tareas más complejas\n")
        
        f.write("\n=== LIMITACIONES DEL ANÁLISIS ===\n")
        f.write("• El conteo de bugs depende de la habilidad de detección del desarrollador\n")
        f.write("• No todos los bugs tienen la misma criticidad (mayor-menor)\n")
        f.write("• Correlación no implica causalidad: factores no medidos pueden influir\n")
        f.write("• Dataset limitado a 500 registros, puede no capturar patrones raros\n")
        f.write("• No se controla por complejidad inherente de las tareas\n")
        f.write("• Factores externos (herramientas de testing, pair programming) no están incluidos\n")
    
    # Mostrar resumen en consola
    print("\n=== ESTADÍSTICAS DE BUGS REPORTADOS ===")
    print(f"Correlación con task_success: {correlacion_bugs:.3f}")
    print(f"Correlación con commits: {correlacion_commits_bugs:.3f}")
    print(f"Correlación con horas: {correlacion_horas_bugs:.3f}")
    
    print(f"\nMediana de bugs: {stats_bugs['mediana']}")
    print(f"Registros con 0 bugs: {porcentaje_cero:.1f}%")
    
    print("\n=== ANÁLISIS POR NÚMERO EXACTO DE BUGS ===")
    for _, row in distribucion_bugs_pd.iterrows():
        print(f"{int(row['bugs_reported'])} bugs: {row['tasa_exito']:.1%} éxito ({row['frecuencia']} registros)")
    
    print("\n=== INSIGHTS CLAVE ===")
    if abs(correlacion_bugs) < 0.1:
        print("• La correlación es prácticamente nula - los bugs no predicen el éxito")
    else:
        print(f"• Existe una correlación {('negativa' if correlacion_bugs < 0 else 'positiva')} de {correlacion_bugs:.3f}")
    
    if porcentaje_cero > 50:
        print("• La mayoría de las sesiones no reportan bugs")
        print("  - Podría indicar tareas simples, falta de detección, o código de alta calidad")
    
    print(f"• Commits por bug (con bugs): {productividad_bugs['avg_commits_per_bug']:.2f}")
    if abs(correlacion_commits_bugs) > 0.3:
        print("• Más commits tienden a asociarse con más bugs (mayor complejidad)")
    elif correlacion_commits_bugs < -0.3:
        print("• Más commits tienden a asociarse con menos bugs (mejor calidad)")
    else:
        print("• No hay relación clara entre commits y bugs")
    
    print(f"\n🎯 VEREDICTO: {veredicto}")
    print(f"📊 Correlación observada: {correlacion_bugs:.3f} (esperada: -0.18)")
    
    print("\n✅ Plan 4 completado exitosamente")
    print("📁 Archivos guardados en notebooks/results/plan4-bugs-reportados/")
    print("   - plan4_bugs_boxplot.png")
    print("   - plan4_bugs_tasa_exito.png") 
    print("   - plan4_bugs_scatter.png")
    print("   - plan4_bugs_categoria.png")
    print("   - plan4_bugs_estadisticas.txt")

if __name__ == "__main__":
    main()
