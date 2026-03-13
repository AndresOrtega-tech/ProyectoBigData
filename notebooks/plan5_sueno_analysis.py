#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan 5: Hipótesis Sueño - Impacto del Descanso en el Éxito
Determinar si dormir menos de ciertas horas impacta negativamente el éxito e identificar el punto óptimo de descanso.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("🧪 Iniciando Plan 5: Análisis de Hipótesis Sueño")
    
    # Paso 1 — Configuración inicial
    print("\n📊 Paso 1: Configuración inicial")
    
    spark = SparkSession.builder \
        .appName("Hipotesis_Sueno") \
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
    
    # 2.1 Promedio de sueño por task_success
    print("🔍 2.1 Promedio de sueño por task_success")
    promedio_sueno = df_spark.groupBy("task_success") \
        .agg(
            avg("sleep_hours").alias("avg_sleep"),
            avg("hours_coding").alias("avg_coding"),
            avg("cognitive_load").alias("avg_cognitive_load"),
            count("*").alias("total_registros")
        )
    
    promedio_sueno.show()
    
    # 2.2 Calcular correlación principal
    print("🔍 2.2 Calcular correlaciones principales")
    correlacion_sueno = df_spark.stat.corr("sleep_hours", "task_success")
    print(f"Correlación sleep_hours vs task_success: {correlacion_sueno:.3f}")
    
    correlacion_sueno_horas = df_spark.stat.corr("sleep_hours", "hours_coding")
    correlacion_sueno_cognitiva = df_spark.stat.corr("sleep_hours", "cognitive_load")
    correlacion_sueno_cafeina = df_spark.stat.corr("sleep_hours", "coffee_intake_mg")
    
    print(f"Correlación sleep_hours vs hours_coding: {correlacion_sueno_horas:.3f}")
    print(f"Correlación sleep_hours vs cognitive_load: {correlacion_sueno_cognitiva:.3f}")
    print(f"Correlación sleep_hours vs coffee_intake_mg: {correlacion_sueno_cafeina:.3f}")
    
    # 2.3 Crear rangos de sueño
    print("🔍 2.3 Crear rangos de sueño")
    df_con_rangos = df_spark.withColumn(
        "sleep_rango",
        when(col("sleep_hours") < 5, "insuficiente")
        .when((col("sleep_hours") >= 5) & (col("sleep_hours") < 6), "bajo")
        .when((col("sleep_hours") >= 6) & (col("sleep_hours") < 8), "optimo")
        .otherwise("alto")
    )
    
    tasa_exito_rango = df_con_rangos.groupBy("sleep_rango") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("hours_coding").alias("avg_hours"),
            avg("cognitive_load").alias("avg_cognitive_load"),
            avg("commits").alias("avg_commits"),
            count("*").alias("total_registros")
        ) \
        .orderBy("tasa_exito", ascending=False)
    
    tasa_exito_rango.show()
    
    # 2.4 Feature engineering: sleep_deficit
    print("🔍 2.4 Análisis de déficit de sueño")
    df_con_deficit = df_con_rangos.withColumn(
        "sleep_deficit",
        when(col("sleep_hours") < 8, 8 - col("sleep_hours")).otherwise(0)
    )
    
    correlacion_deficit = df_con_deficit.stat.corr("sleep_deficit", "task_success")
    print(f"Correlación sleep_deficit vs task_success: {correlacion_deficit:.3f}")
    
    df_con_deficit = df_con_deficit.withColumn(
        "deficit_rango",
        when(col("sleep_deficit") == 0, "sin_deficit")
        .when((col("sleep_deficit") > 0) & (col("sleep_deficit") <= 1), "deficit_leve")
        .when((col("sleep_deficit") > 1) & (col("sleep_deficit") <= 2), "deficit_moderado")
        .otherwise("deficit_severo")
    )
    
    deficit_stats = df_con_deficit.groupBy("deficit_rango") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("sleep_hours").alias("avg_sleep"),
            avg("cognitive_load").alias("avg_cognitive_load"),
            count("*").alias("total_registros")
        ) \
        .orderBy("tasa_exito", ascending=False)
    
    deficit_stats.show()
    
    # 2.5 Análisis por hora exacta
    print("🔍 2.5 Análisis por hora exacta de sueño")
    horas_exito = df_spark.groupBy("sleep_hours") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            count("*").alias("count")
        ) \
        .filter(col("count") >= 10) \
        .orderBy("sleep_hours")
    
    horas_exito.show()
    
    # 2.6 Análisis de interacción: sueño + horas de código
    print("🔍 2.6 Análisis de interacción sueño + horas de código")
    df_con_interaccion = df_con_deficit.withColumn(
        "hours_rango",
        when(col("hours_coding") < 3, "pocas")
        .when((col("hours_coding") >= 3) & (col("hours_coding") < 6), "moderadas")
        .when((col("hours_coding") >= 6) & (col("hours_coding") < 9), "muchas")
        .otherwise("excesivas")
    )
    
    interaccion = df_con_interaccion.groupBy("sleep_rango", "hours_rango") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("cognitive_load").alias("avg_cognitive_load"),
            count("*").alias("total_registros")
        ) \
        .filter(col("total_registros") >= 5) \
        .orderBy("tasa_exito", ascending=False)
    
    interaccion.show()
    
    # Paso 3 — Convertir a pandas para visualización
    print("\n🔄 Paso 3: Convertir a pandas para visualización")
    
    promedio_sueno_pd = promedio_sueno.toPandas()
    tasa_exito_rango_pd = tasa_exito_rango.toPandas()
    deficit_stats_pd = deficit_stats.toPandas()
    horas_exito_pd = horas_exito.toPandas()
    interaccion_pd = interaccion.toPandas()
    df_completo_pd = df_spark.toPandas()
    
    spark.stop()
    print("✅ Sesión Spark detenida")
    
    # Paso 4 — Visualizaciones
    print("\n📊 Paso 4: Generando visualizaciones")
    
    # Crear directorio de resultados
    os.makedirs("notebooks/results/plan5-sueno", exist_ok=True)
    
    # Gráfico 1: Boxplot de horas de sueño por task_success
    print("📈 Gráfico 1: Distribución de sueño por éxito")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_completo_pd, x="task_success", y="sleep_hours")
    plt.title("Hipótesis Sueño: Distribución de Horas de Sueño por Éxito", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Éxito de Tarea (0=Fracaso, 1=Éxito)", fontsize=12)
    plt.ylabel("Horas de Sueño", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Agregar medianas
    for i, success in enumerate([0, 1]):
        subset = df_completo_pd[df_completo_pd['task_success'] == success]
        median = subset['sleep_hours'].median()
        plt.text(i, median + 0.1, f"Mediana: {median:.1f}h", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Agregar línea de referencia para 8 horas
    plt.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='8 horas (referencia)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan5-sueno/plan5_sueno_boxplot.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 2: Barplot de tasa de éxito por rango de sueño
    print("📈 Gráfico 2: Tasa de éxito por rango de sueño")
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(data=tasa_exito_rango_pd, x="sleep_rango", y="tasa_exito")
    plt.title("Hipótesis Sueño: Tasa de Éxito por Nivel de Descanso", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Nivel de Sueño", fontsize=12)
    plt.ylabel("Tasa de Éxito", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Agregar etiquetas de porcentaje y número de desarrolladores
    for i, row in tasa_exito_rango_pd.iterrows():
        ax.text(i, row.tasa_exito + 0.02, f"{row.tasa_exito:.1%}\n({row.total_registros} devs)", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Agregar descripción de rangos en el eje X
    rango_labels = {
        'insuficiente': '< 5h\n(Déficit severo)',
        'bajo': '5-6h\n(Déficit leve)',
        'optimo': '6-8h\n(Rango recomendado)',
        'alto': '> 8h\n(Sueño excesivo)'
    }
    
    ax = plt.gca()
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    new_labels = [rango_labels.get(label, label) for label in current_labels]
    ax.set_xticks(range(len(current_labels)))
    ax.set_xticklabels(new_labels)
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan5-sueno/plan5_sueno_tasa_exito.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 3: Histograma de horas de sueño coloreado por task_success
    print("📈 Gráfico 3: Distribución de horas de sueño")
    plt.figure(figsize=(14, 8))
    sns.histplot(data=df_completo_pd, x="sleep_hours", hue="task_success", 
                 bins=15, alpha=0.7, kde=True)
    plt.title("Hipótesis Sueño: Distribución de Horas de Sueño por Éxito", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Horas de Sueño", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.legend(title="Éxito", labels=["Éxito", "Fracaso"])
    plt.grid(True, alpha=0.3)
    
    # Agregar líneas de referencia para rangos
    plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Límite < 5h')
    plt.axvline(x=6, color='green', linestyle='--', alpha=0.7, label='Límite < 6h')
    plt.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Límite < 8h')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan5-sueno/plan5_sueno_histograma.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 4: Línea de tendencia de éxito por horas exactas de sueño
    print("📈 Gráfico 4: Tasa de éxito por horas exactas")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=horas_exito_pd, x="sleep_hours", y="tasa_exito", 
                 marker='o', markersize=8, linewidth=2)
    plt.title("Hipótesis Sueño: Tasa de Éxito por Horas Exactas de Descanso", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Horas de Sueño", fontsize=12)
    plt.ylabel("Tasa de Éxito", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Agregar línea de referencia para 8 horas
    plt.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='8 horas (referencia)')
    
    # Encontrar y marcar el punto óptimo
    punto_optimo = horas_exito_pd.loc[horas_exito_pd['tasa_exito'].idxmax()]
    plt.scatter(punto_optimo['sleep_hours'], punto_optimo['tasa_exito'], 
               color='green', s=200, zorder=5, label=f'Óptimo: {punto_optimo["sleep_hours"]:.1f}h')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/results/plan5-sueno/plan5_sueno_linea.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 5: Heatmap de interacción sueño + horas de código
    print("📈 Gráfico 5: Interacción sueño + horas de código")
    # Pivot para heatmap
    heatmap_data = interaccion_pd.pivot(index="sleep_rango", 
                                      columns="hours_rango", 
                                      values="tasa_exito")
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", 
               vmin=0, vmax=1, cbar_kws={'label': 'Tasa de Éxito'})
    plt.title("Hipótesis Sueño: Tasa de Éxito por Combinación Descanso + Trabajo", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Rango de Horas de Código", fontsize=12)
    plt.ylabel("Nivel de Sueño", fontsize=12)
    plt.tight_layout()
    plt.savefig("notebooks/results/plan5-sueno/plan5_sueno_heatmap.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Paso 5 — Análisis de resultados
    print("\n📋 Paso 5: Análisis de resultados")
    
    # Guardar estadísticas en archivo
    with open("notebooks/results/plan5-sueno/plan5_sueno_estadisticas.txt", "w", encoding='utf-8') as f:
        f.write("=== PLAN 5: ANÁLISIS DE HIPÓTESIS SUEÑO ===\n")
        f.write("Hipótesis: 'Dormir menos de X horas impacta negativamente el task_success'\n")
        f.write("Correlación: por confirmar\n\n")
        
        f.write("=== METODOLOGÍA ===\n")
        f.write("• Dataset: 500 registros de productividad de desarrolladores\n")
        f.write("• Variable independiente: sleep_hours (horas de sueño la noche anterior)\n")
        f.write("• Variable dependiente: task_success (0=fallo, 1=éxito)\n")
        f.write("• Análisis: Correlación de Pearson + rangos + déficit + interacción\n")
        f.write("• Herramientas: PySpark para procesamiento, pandas/matplotlib para visualización\n\n")
        
        f.write("=== DEFINICIÓN DE RANGOS DE SUEÑO ===\n")
        f.write("• Rango INSUFICIENTE: < 5 horas (déficit severo, impacto significativo)\n")
        f.write("• Rango BAJO: 5-6 horas (déficit leve, ligero impacto)\n")
        f.write("• Rango ÓPTIMO: 6-8 horas (rango recomendado por expertos)\n")
        f.write("• Rango ALTO: > 8 horas (sueño excesivo, posible impacto negativo)\n")
        f.write("• Referencia: Recomendación NSF de 7-9 horas para adultos\n\n")
        
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        f.write(f"Correlación de Pearson: {correlacion_sueno:.3f}\n")
        f.write("Interpretación de correlación:\n")
        f.write("• 0.0-0.1: Correlación nula o insignificante\n")
        f.write("• 0.1-0.3: Correlación débil\n")
        f.write("• 0.3-0.5: Correlación moderada\n")
        f.write("• 0.5-0.7: Correlación fuerte\n")
        f.write("• 0.7-1.0: Correlación muy fuerte\n\n")
        
        f.write("Análisis por grupo de éxito:\n\n")
        
        for success in [0, 1]:
            subset = df_completo_pd[df_completo_pd['task_success'] == success]
            grupo = "FRACASO" if success == 0 else "ÉXITO"
            f.write(f"Grupo {grupo} (task_success = {success}):\n")
            f.write(f"  • Promedio sueño: {subset['sleep_hours'].mean():.1f} h\n")
            f.write(f"  • Mediana sueño: {subset['sleep_hours'].median():.1f} h\n")
            f.write(f"  • Desviación estándar: {subset['sleep_hours'].std():.1f} h\n")
            f.write(f"  • Mínimo: {subset['sleep_hours'].min():.1f} h\n")
            f.write(f"  • Máximo: {subset['sleep_hours'].max():.1f} h\n")
            f.write(f"  • Total registros: {len(subset)}\n\n")
        
        # Calcular diferencia clave
        grupo_exito = df_completo_pd[df_completo_pd['task_success'] == 1]
        grupo_fracaso = df_completo_pd[df_completo_pd['task_success'] == 0]
        diff_sueno = grupo_exito['sleep_hours'].mean() - grupo_fracaso['sleep_hours'].mean()
        
        f.write("Diferencia clave:\n")
        f.write(f"  • Los desarrolladores con éxito duermen {diff_sueno:.1f} horas más en promedio\n")
        f.write(f"  • Esto representa un {(diff_sueno/grupo_fracaso['sleep_hours'].mean()*100):.1f}% más de descanso\n\n")
        
        f.write("=== ANÁLISIS POR RANGOS DE SUEÑO ===\n")
        f.write("Tasa de éxito y características por rango:\n\n")
        
        for _, row in tasa_exito_rango_pd.iterrows():
            f.write(f"Rango {row['sleep_rango'].upper()}:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Horas de coding promedio: {row['avg_hours']:.1f} h\n")
            f.write(f"  • Carga cognitiva promedio: {row['avg_cognitive_load']:.1f}\n")
            f.write(f"  • Commits promedio: {row['avg_commits']:.1f}\n")
            f.write(f"  • Total registros: {row['total_registros']}\n")
            f.write(f"  • Proporción del dataset: {(row['total_registros']/len(df_completo_pd)*100):.1f}%\n")
            
            # Interpretación
            if row['sleep_rango'] == 'insuficiente':
                interpretacion = "Descanso inadecuado que afecta rendimiento cognitivo"
            elif row['sleep_rango'] == 'bajo':
                interpretacion = "Descanso mínimo con ligero impacto en rendimiento"
            elif row['sleep_rango'] == 'optimo':
                interpretacion = "Descanso adecuado para máximo rendimiento"
            else:  # alto
                interpretacion = "Sueño excesivo que puede indicar otros problemas"
            
            f.write(f"  • Interpretación: {interpretacion}\n\n")
        
        f.write("=== ANÁLISIS POR DÉFICIT DE SUEÑO ===\n")
        f.write("Impacto del déficit respecto a 8 horas recomendadas:\n\n")
        
        for _, row in deficit_stats_pd.iterrows():
            f.write(f"Rango {row['deficit_rango'].replace('_', ' ').upper()}:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Sueño promedio: {row['avg_sleep']:.1f} h\n")
            f.write(f"  • Carga cognitiva: {row['avg_cognitive_load']:.1f}\n")
            f.write(f"  • Total registros: {row['total_registros']}\n\n")
        
        f.write("=== ANÁLISIS POR HORA EXACTA ===\n")
        f.write("Patrones detallados por horas específicas:\n\n")
        
        for _, row in horas_exito_pd.iterrows():
            hora = row['sleep_hours']
            f.write(f"{hora:.1f} horas de sueño:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Frecuencia: {row['count']} registros\n")
            
            # Interpretación por hora
            if hora < 5:
                interpretacion = "Déficit severo, alto riesgo de fracaso"
            elif hora < 6:
                interpretacion = "Déficit leve, rendimiento reducido"
            elif hora <= 8:
                interpretacion = "Rango óptimo, máximo rendimiento"
            else:
                interpretacion = "Sueño excesivo, posible fatiga"
            
            f.write(f"  • Interpretación: {interpretacion}\n\n")
        
        f.write("=== INSIGHTS CLAVE ===\n")
        f.write("1. RELACIÓN SUEÑO-ÉXITO:\n")
        if correlacion_sueno > 0.3:
            f.write(f"   • Correlación moderada-fuerte ({correlacion_sueno:.3f}) indica relación positiva clara\n")
        elif correlacion_sueno > 0.1:
            f.write(f"   • Correlación débil ({correlacion_sueno:.3f}) indica relación positiva limitada\n")
        elif correlacion_sueno < -0.1:
            f.write(f"   • Correlación negativa inesperada ({correlacion_sueno:.3f})\n")
        else:
            f.write(f"   • Correlación prácticamente nula ({correlacion_sueno:.3f})\n")
        
        f.write("2. PUNTO DE CORTE ÓPTIMO:\n")
        if not horas_exito_pd.empty:
            punto_optimo = horas_exito_pd.loc[horas_exito_pd['tasa_exito'].idxmax()]
            f.write(f"   • Horas óptimas: {punto_optimo['sleep_hours']:.1f} horas\n")
            f.write(f"   • Tasa de éxito máxima: {punto_optimo['tasa_exito']:.1%}\n")
        
        f.write("3. IMPACTO DEL DÉFICIT:\n")
        if correlacion_deficit < -0.1:
            f.write(f"   • Déficit de sueño reduce éxito (r={correlacion_deficit:.3f})\n")
        else:
            f.write(f"   • Déficit de sueño tiene impacto limitado (r={correlacion_deficit:.3f})\n")
        
        f.write("4. EXCESO DE SUEÑO:\n")
        rango_alto = tasa_exito_rango_pd[tasa_exito_rango_pd['sleep_rango'] == 'alto']
        if not rango_alto.empty:
            tasa_alta = rango_alto['tasa_exito'].iloc[0]
            max_idx = tasa_exito_rango_pd['tasa_exito'].idxmax()
            rango_optimo = tasa_exito_rango_pd.loc[max_idx]
            tasa_optima = rango_optimo['tasa_exito']
            if tasa_alta < tasa_optima * 0.9:
                f.write(f"   • Dormir demasiado reduce éxito en {(1-tasa_alta/tasa_optima)*100:.0f}%\n")
            else:
                f.write("   • No hay evidencia de que dormir demasiado sea perjudicial\n")
        
        f.write("\n=== COMPARACIÓN DE IMPORTANCIA ===\n")
        f.write("Impacto relativo en task_success:\n")
        f.write("• Cafeína vs Éxito: ~0.70 (muy fuerte positivo)\n")
        f.write("• Horas vs Éxito: ~0.62 (fuerte positivo)\n")
        f.write("• Carga Cognitiva vs Éxito: ~-0.20 (débil negativo)\n")
        f.write("• Bugs vs Éxito: ~-0.18 (muy débil negativo)\n")
        f.write(f"• Sueño vs Éxito: {correlacion_sueno:.3f} (débil a moderado)\n")
        f.write("\nConclusión: El sueño tiene un impacto moderado en el éxito\n")
        
        f.write("\n=== VEREDICTO FINAL ===\n")
        if correlacion_sueno > 0.3:
            veredicto = "HIPÓTESIS CONFIRMADA"
            f.write(f"✅ {veredicto}\n")
            f.write("Evidencia fuerte que apoya la hipótesis original\n")
        elif correlacion_sueno > 0.1:
            veredicto = "HIPÓTESIS CONFIRMADA (débil)"
            f.write(f"✅ {veredicto}\n")
            f.write("Evidencia moderada que apoya la hipótesis original\n")
        elif correlacion_sueno < -0.1:
            veredicto = "HIPÓTESIS REFUTADA"
            f.write(f"❌ {veredicto}\n")
            f.write("Evidencia contraria a la hipótesis original\n")
        else:
            veredicto = "HIPÓTESIS NEUTRA"
            f.write(f"🔄 {veredicto}\n")
            f.write("Evidencia insuficiente para confirmar o refutar\n")
        
        f.write("\nMétricas clave:\n")
        f.write(f"• Correlación observada: {correlacion_sueno:.3f}\n")
        f.write(f"• Correlación con déficit: {correlacion_deficit:.3f}\n")
        if not horas_exito_pd.empty:
            max_idx = horas_exito_pd['tasa_exito'].idxmax()
            punto_optimo = horas_exito_pd.loc[max_idx]
            f.write(f"• Horas óptimas: {punto_optimo['sleep_hours']:.1f} h\n")
        
        f.write("\n=== RECOMENDACIONES PRÁCTICAS ===\n")
        if not horas_exito_pd.empty:
            horas_optimas = punto_optimo['sleep_hours']
            f.write(f"• Dormir {horas_optimas:.1f} horas para maximizar éxito\n")
            
            if horas_optimas >= 6 and horas_optimas <= 8:
                f.write("• Este rango coincide con recomendaciones científicas (6-8 horas)\n")
            elif horas_optimas < 6:
                f.write("• Sorprendentemente, menos de 6 horas parece óptimo en este dataset\n")
            else:
                f.write("• Más de 8 horas parece óptimo, posiblemente por factores específicos\n")
        
        if correlacion_deficit < -0.1:
            f.write("• Evitar déficit de sueño (>2 horas menos que 8 horas recomendadas)\n")
            f.write("• Priorizar descanso para tareas críticas\n")
        
        f.write("\n=== LIMITACIONES DEL ANÁLISIS ===\n")
        f.write("• Horas de sueño son auto-reportadas, pueden no ser precisas\n")
        f.write("• Correlación no implica causalidad: otros factores pueden influir\n")
        f.write("• Dataset limitado a 500 registros, puede no capturar toda la variabilidad\n")
        f.write("• No se controla por calidad del sueño (profundidad, interrupciones)\n")
        f.write("• Factores individuales (cronotipo, edad, salud) no están incluidos\n")
        f.write("• Relación puede variar según tipo de tareas y presión temporal\n")
    
    # Mostrar resumen en consola
    print("\n=== ESTADÍSTICAS DE SUEÑO ===")
    print(f"Correlación con task_success: {correlacion_sueno:.3f}")
    print(f"Correlación con sleep_deficit: {correlacion_deficit:.3f}")
    
    for success in [0, 1]:
        subset = df_completo_pd[df_completo_pd['task_success'] == success]
        print(f"\nTask Success = {success}:")
        print(f"  Promedio sueño: {subset['sleep_hours'].mean():.1f} h")
        print(f"  Mediana sueño: {subset['sleep_hours'].median():.1f} h")
        print(f"  Desviación estándar: {subset['sleep_hours'].std():.1f} h")
    
    print("\n=== ANÁLISIS POR RANGO DE SUEÑO ===")
    for _, row in tasa_exito_rango_pd.iterrows():
        print(f"Rango {row['sleep_rango']}: {row['tasa_exito']:.1%} de éxito ({row['total_registros']} registros)")
    
    print("\n=== ANÁLISIS POR DÉFICIT DE SUEÑO ===")
    for _, row in deficit_stats_pd.iterrows():
        print(f"Rango {row['deficit_rango']}: {row['tasa_exito']:.1%} de éxito ({row['total_registros']} registros)")
    
    # Identificar punto de corte
    if not tasa_exito_rango_pd.empty:
        max_idx = tasa_exito_rango_pd['tasa_exito'].idxmax()
        rango_optimo = tasa_exito_rango_pd.loc[max_idx]
        mejor_rango = rango_optimo['sleep_rango']
        mejor_tasa = rango_optimo['tasa_exito']
        
        print(f"\n=== PUNTO DE CORTE IDENTIFICADO ===")
        print(f"Mejor rango: '{mejor_rango}' con {mejor_tasa:.1%} de éxito")
        
        # Definir recomendación basada en resultados
        if mejor_rango == "optimo":
            punto_corte = "6-8 horas"
        elif mejor_rango == "alto":
            punto_corte = "más de 8 horas"
        else:
            punto_corte = mejor_rango
        
        print(f"RECOMENDACIÓN: Dormir {punto_corte} para maximizar el éxito")
        
        # Análisis de exceso
        if "alto" in tasa_exito_rango_pd['sleep_rango'].values:
            tasa_alto = tasa_exito_rango_pd[tasa_exito_rango_pd['sleep_rango'] == 'alto']['tasa_exito'].iloc[0]
            if tasa_alto < mejor_tasa * 0.9:
                print(f"⚠️  ALERTA: Dormir demasiado reduce el éxito en {(1-tasa_alto/mejor_tasa)*100:.0f}%")
    
    print(f"\n🎯 VEREDICTO: {veredicto}")
    print(f"📊 Correlación observada: {correlacion_sueno:.3f}")
    print(f"📊 Correlación con déficit: {correlacion_deficit:.3f}")
    
    print("\n✅ Plan 5 completado exitosamente")
    print("📁 Archivos guardados en notebooks/results/plan5-sueno/")
    print("   - plan5_sueno_boxplot.png")
    print("   - plan5_sueno_tasa_exito.png") 
    print("   - plan5_sueno_histograma.png")
    print("   - plan5_sueno_linea.png")
    print("   - plan5_sueno_heatmap.png")
    print("   - plan5_sueno_estadisticas.txt")

if __name__ == "__main__":
    main()
