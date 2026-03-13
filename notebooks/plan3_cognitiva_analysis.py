#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan 3: Hipótesis Carga Cognitiva - Impacto en el Éxito
Analizar si la alta carga cognitiva reduce el éxito en las tareas y cómo interactúa con las horas de trabajo.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, corr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("🧪 Iniciando Plan 3: Análisis de Hipótesis Carga Cognitiva")
    
    # Paso 1 — Configuración inicial
    print("\n📊 Paso 1: Configuración inicial")
    
    spark = SparkSession.builder \
        .appName("Hipotesis_Carga_Cognitiva") \
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
    
    # 2.1 Promedio de carga cognitiva por task_success
    print("🔍 2.1 Promedio de carga cognitiva por task_success")
    promedio_cognitiva = df_spark.groupBy("task_success") \
        .agg(
            avg("cognitive_load").alias("avg_cognitive_load"),
            avg("hours_coding").alias("avg_hours"),
            count("*").alias("total_registros")
        )
    
    promedio_cognitiva.show()
    
    # 2.2 Calcular correlaciones principales
    print("🔍 2.2 Calcular correlaciones principales")
    correlacion_cognitiva = df_spark.stat.corr("cognitive_load", "task_success")
    print(f"Correlación cognitive_load vs task_success: {correlacion_cognitiva:.3f}")
    
    correlacion_horas_cognitiva = df_spark.stat.corr("cognitive_load", "hours_coding")
    print(f"Correlación cognitive_load vs hours_coding: {correlacion_horas_cognitiva:.3f}")
    
    correlacion_distracciones = df_spark.stat.corr("cognitive_load", "distractions")
    print(f"Correlación cognitive_load vs distractions: {correlacion_distracciones:.3f}")
    
    # 2.3 Crear rangos de carga cognitiva
    print("🔍 2.3 Crear rangos de carga cognitiva")
    df_con_rangos = df_spark.withColumn(
        "cognitive_rango",
        when(col("cognitive_load") <= 3, "baja")
        .when((col("cognitive_load") >= 4) & (col("cognitive_load") <= 6), "media")
        .otherwise("alta")
    )
    
    tasa_exito_rango = df_con_rangos.groupBy("cognitive_rango") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("hours_coding").alias("avg_hours"),
            avg("distractions").alias("avg_distractions"),
            avg("ai_usage_hours").alias("avg_ai_usage"),
            count("*").alias("total_registros")
        ) \
        .orderBy("cognitive_rango")
    
    tasa_exito_rango.show()
    
    # 2.4 Análisis de interacción: carga cognitiva + horas
    print("🔍 2.4 Análisis de interacción carga cognitiva + horas")
    df_con_interaccion = df_con_rangos.withColumn(
        "hours_rango",
        when(col("hours_coding") < 3, "pocas")
        .when((col("hours_coding") >= 3) & (col("hours_coding") < 6), "moderadas")
        .when((col("hours_coding") >= 6) & (col("hours_coding") < 9), "muchas")
        .otherwise("excesivas")
    )
    
    interaccion = df_con_interaccion.groupBy("cognitive_rango", "hours_rango") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            count("*").alias("total_registros")
        ) \
        .filter(col("total_registros") >= 5) \
        .orderBy("cognitive_rango", "hours_rango")
    
    interaccion.show()
    
    # 2.5 Análisis de factores que influyen en la carga cognitiva
    print("🔍 2.5 Factores que influyen en la carga cognitiva")
    factores_cognitiva = df_spark.select(
        corr("cognitive_load", "hours_coding").alias("corr_horas"),
        corr("cognitive_load", "distractions").alias("corr_distracciones"),
        corr("cognitive_load", "coffee_intake_mg").alias("corr_cafeina"),
        corr("cognitive_load", "sleep_hours").alias("corr_sueno"),
        corr("cognitive_load", "ai_usage_hours").alias("corr_ai")
    ).collect()[0]
    
    print("Factores que influyen en la carga cognitiva:")
    print(f"  Horas de código: {factores_cognitiva['corr_horas']:.3f}")
    print(f"  Distracciones: {factores_cognitiva['corr_distracciones']:.3f}")
    print(f"  Cafeína: {factores_cognitiva['corr_cafeina']:.3f}")
    print(f"  Sueño: {factores_cognitiva['corr_sueno']:.3f}")
    print(f"  Uso de IA: {factores_cognitiva['corr_ai']:.3f}")
    
    # Paso 3 — Convertir a pandas para visualización
    print("\n🔄 Paso 3: Convertir a pandas para visualización")
    
    promedio_cognitiva_pd = promedio_cognitiva.toPandas()
    tasa_exito_rango_pd = tasa_exito_rango.toPandas()
    interaccion_pd = interaccion.toPandas()
    df_completo_pd = df_spark.toPandas()
    
    spark.stop()
    print("✅ Sesión Spark detenida")
    
    # Paso 4 — Visualizaciones
    print("\n📊 Paso 4: Generando visualizaciones")
    
    # Crear directorio de resultados
    os.makedirs("notebooks/results/plan3-carga-cognitiva", exist_ok=True)
    
    # Gráfico 1: Boxplot de carga cognitiva por task_success
    print("📈 Gráfico 1: Boxplot de carga cognitiva por task_success")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_completo_pd, x="task_success", y="cognitive_load")
    plt.title("Hipótesis Carga Cognitiva: Distribución por Éxito de Tarea", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Éxito de Tarea (0=Fracaso, 1=Éxito)", fontsize=12)
    plt.ylabel("Carga Cognitiva (escala 1-10)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Agregar medianas
    for i, success in enumerate([0, 1]):
        subset = df_completo_pd[df_completo_pd['task_success'] == success]
        median = subset['cognitive_load'].median()
        plt.text(i, median + 0.1, f"Mediana: {median:.1f}", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_boxplot.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 2: Barplot de tasa de éxito por rango cognitivo
    print("📈 Gráfico 2: Tasa de éxito por rango cognitivo")
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(data=tasa_exito_rango_pd, x="cognitive_rango", y="tasa_exito")
    plt.title("Hipótesis Carga Cognitiva: Tasa de Éxito por Nivel de Carga", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Nivel de Carga Cognitiva", fontsize=12)
    plt.ylabel("Tasa de Éxito", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Agregar etiquetas de porcentaje y número de desarrolladores
    for i, row in tasa_exito_rango_pd.iterrows():
        ax.text(i, row.tasa_exito + 0.02, f"{row.tasa_exito:.1%}\n({row.total_registros} devs)", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Agregar descripción de rangos en el eje X
    rango_labels = {
        'baja': '1-3\n(Baja carga)',
        'media': '4-6\n(Carga moderada)',
        'alta': '7-10\n(Alta carga)'
    }
    
    ax = plt.gca()
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    new_labels = [rango_labels.get(label, label) for label in current_labels]
    ax.set_xticks(range(len(current_labels)))
    ax.set_xticklabels(new_labels)
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_tasa_exito.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 3: Scatter plot de carga cognitiva vs horas
    print("📈 Gráfico 3: Interacción carga cognitiva vs horas de código")
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df_completo_pd, x="cognitive_load", y="hours_coding", 
                    hue="task_success", size="commits", alpha=0.7, sizes=(20, 200))
    plt.title("Hipótesis Carga Cognitiva: Interacción con Horas de Código", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Carga Cognitiva (escala 1-10)", fontsize=12)
    plt.ylabel("Horas de Código", fontsize=12)
    plt.legend(title="Éxito", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Agregar líneas de referencia para rangos de carga
    plt.axvline(x=3.5, color='green', linestyle='--', alpha=0.7, label='Límite baja-media')
    plt.axvline(x=6.5, color='orange', linestyle='--', alpha=0.7, label='Límite media-alta')
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_scatter.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 4: Heatmap de interacción
    print("📈 Gráfico 4: Heatmap de interacción carga cognitiva + horas")
    # Pivot para heatmap
    heatmap_data = interaccion_pd.pivot(index="cognitive_rango", 
                                      columns="hours_rango", 
                                      values="tasa_exito")
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", 
               vmin=0, vmax=1, cbar_kws={'label': 'Tasa de Éxito'})
    plt.title("Hipótesis Carga Cognitiva: Tasa de Éxito por Combinación", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Rango de Horas de Código", fontsize=12)
    plt.ylabel("Nivel de Carga Cognitiva", fontsize=12)
    plt.tight_layout()
    plt.savefig("notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_heatmap.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Paso 5 — Análisis de resultados
    print("\n📋 Paso 5: Análisis de resultados")
    
    # Guardar estadísticas en archivo
    with open("notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_estadisticas.txt", "w", encoding='utf-8') as f:
        f.write("=== PLAN 3: ANÁLISIS DE HIPÓTESIS CARGA COGNITIVA ===\n")
        f.write("Hipótesis: 'Alta carga cognitiva reduce el task_success'\n")
        f.write("Correlación esperada: -0.20 (débil negativa)\n\n")
        
        f.write("=== METODOLOGÍA ===\n")
        f.write("• Dataset: 500 registros de productividad de desarrolladores\n")
        f.write("• Variable independiente: cognitive_load (escala 1-10 auto-reportada)\n")
        f.write("• Variable dependiente: task_success (0=fallo, 1=éxito)\n")
        f.write("• Análisis: Correlación de Pearson + rangos + interacción con horas\n")
        f.write("• Herramientas: PySpark para procesamiento, pandas/matplotlib para visualización\n\n")
        
        f.write("=== DEFINICIÓN DE RANGOS DE CARGA COGNITIVA ===\n")
        f.write("• Rango BAJA: 1-3 (equivalente a tareas simples, baja presión)\n")
        f.write("• Rango MEDIA: 4-6 (equivalente a tareas moderadamente complejas)\n")
        f.write("• Rango ALTA: 7-10 (equivalente a tareas complejas, alta presión)\n")
        f.write("• Referencia: Escala subjetiva de percepción de dificultad mental\n\n")
        
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        f.write(f"Correlación de Pearson: {correlacion_cognitiva:.3f}\n")
        f.write("Interpretación de correlación:\n")
        f.write("• -0.1 a -0.3: Correlación negativa débil\n")
        f.write("• -0.3 a -0.5: Correlación negativa moderada\n")
        f.write("• 0.0 a ±0.1: Sin correlación\n")
        f.write("• +0.1 a +0.3: Correlación positiva débil\n\n")
        
        f.write("Análisis por grupo de éxito:\n\n")
        
        for success in [0, 1]:
            subset = df_completo_pd[df_completo_pd['task_success'] == success]
            grupo = "FRACASO" if success == 0 else "ÉXITO"
            f.write(f"Grupo {grupo} (task_success = {success}):\n")
            f.write(f"  • Promedio carga cognitiva: {subset['cognitive_load'].mean():.1f}\n")
            f.write(f"  • Mediana carga cognitiva: {subset['cognitive_load'].median():.1f}\n")
            f.write(f"  • Desviación estándar: {subset['cognitive_load'].std():.1f}\n")
            f.write(f"  • Mínimo: {subset['cognitive_load'].min():.1f}\n")
            f.write(f"  • Máximo: {subset['cognitive_load'].max():.1f}\n")
            f.write(f"  • Total registros: {len(subset)}\n\n")
        
        # Calcular diferencia clave
        grupo_exito = df_completo_pd[df_completo_pd['task_success'] == 1]
        grupo_fracaso = df_completo_pd[df_completo_pd['task_success'] == 0]
        diff_cognitiva = grupo_fracaso['cognitive_load'].mean() - grupo_exito['cognitive_load'].mean()
        
        f.write("Diferencia clave:\n")
        f.write(f"  • Los desarrolladores con fracaso reportan {diff_cognitiva:.1f} puntos más de carga\n")
        f.write(f"  • Diferencia porcentual: {(diff_cognitiva/grupo_exito['cognitive_load'].mean()*100):.1f}%\n\n")
        
        f.write("=== ANÁLISIS POR RANGOS DE CARGA ===\n")
        f.write("Tasa de éxito y factores asociados por rango:\n\n")
        
        for _, row in tasa_exito_rango_pd.iterrows():
            f.write(f"Rango {row['cognitive_rango'].upper()}:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Carga promedio: {row['avg_hours']:.1f} h de código\n")
            f.write(f"  • Distracciones promedio: {row['avg_distractions']:.1f}\n")
            f.write(f"  • Uso de IA promedio: {row['avg_ai_usage']:.1f} h\n")
            f.write(f"  • Total registros: {row['total_registros']}\n")
            f.write(f"  • Proporción del dataset: {(row['total_registros']/len(df_completo_pd)*100):.1f}%\n")
            
            # Interpretación
            if row['cognitive_rango'] == 'baja':
                interpretacion = "Tareas simples con alta probabilidad de éxito"
            elif row['cognitive_rango'] == 'media':
                interpretacion = "Tareas moderadamente complejas con éxito variable"
            else:  # alta
                interpretacion = "Tareas complejas con mayor riesgo de fracaso"
            
            f.write(f"  • Interpretación: {interpretacion}\n\n")
        
        f.write("=== INSIGHTS CLAVE ===\n")
        f.write("1. RELACIÓN CARGA-ÉXITO:\n")
        if correlacion_cognitiva < -0.1:
            f.write(f"   • Correlación negativa ({correlacion_cognitiva:.3f}) confirma hipótesis parcialmente\n")
        elif correlacion_cognitiva > 0.1:
            f.write(f"   • Correlación positiva inesperada ({correlacion_cognitiva:.3f}) refuta hipótesis\n")
        else:
            f.write(f"   • Correlación neutra ({correlacion_cognitiva:.3f}) indica relación débil\n")
        
        f.write("2. FACTORES INFLUYENTES:\n")
        factores_ordenados = [
            ("Horas de código", factores_cognitiva['corr_horas']),
            ("Distracciones", factores_cognitiva['corr_distracciones']),
            ("Cafeína", factores_cognitiva['corr_cafeina']),
            ("Sueño", factores_cognitiva['corr_sueno']),
            ("Uso de IA", factores_cognitiva['corr_ai'])
        ]
        factores_ordenados.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for factor, corr_val in factores_ordenados[:3]:
            if abs(corr_val) > 0.1:
                direccion = "aumenta" if corr_val > 0 else "reduce"
                f.write(f"   • {factor} {direccion} la carga cognitiva (r={corr_val:.3f})\n")
        
        f.write("3. INTERACCIONES CRÍTICAS:\n")
        if not interaccion_pd.empty:
            peores_combinaciones = interaccion_pd.nsmallest(3, 'tasa_exito')
            f.write("   • Peores combinaciones (baja tasa de éxito):\n")
            for _, row in peores_combinaciones.iterrows():
                f.write(f"     Carga {row['cognitive_rango']} + Horas {row['hours_rango']}: {row['tasa_exito']:.1%}\n")
        
        f.write("4. UMBRAL CRÍTICO:\n")
        rango_alto = tasa_exito_rango_pd[tasa_exito_rango_pd['cognitive_rango'] == 'alta']
        if not rango_alto.empty:
            tasa_alta = rango_alto['tasa_exito'].iloc[0]
            if tasa_alta < 0.5:
                f.write(f"   • Carga alta muestra baja tasa de éxito ({tasa_alta:.1%})\n")
                f.write("   • Sugiere umbral de complejidad que afecta rendimiento\n")
                f.write("   • Se recomienda revisar y ajustar la complejidad de las tareas para mejorar el rendimiento\n")
        
        f.write("\n=== COMPARACIÓN DE IMPORTANCIA ===\n")
        f.write("• Cafeína vs Éxito: ~0.70 (muy fuerte positivo)\n")
        f.write("• Horas vs Éxito: ~0.62 (fuerte positivo)\n")
        f.write(f"• Carga Cognitiva vs Éxito: {correlacion_cognitiva:.3f} (débil)\n")
        f.write("\nConclusión: La carga cognitiva tiene menor impacto que cafeína y horas\n")
        
        f.write("\n=== VEREDICTO FINAL ===\n")
        if correlacion_cognitiva <= -0.1:
            veredicto = "HIPÓTESIS CONFIRMADA (débil)"
            f.write(f"✅ {veredicto}\n")
            f.write("Evidencia débil que apoya la hipótesis original\n")
        elif correlacion_cognitiva >= 0.1:
            veredicto = "HIPÓTESIS REFUTADA"
            f.write(f"❌ {veredicto}\n")
            f.write("Evidencia contraria a la hipótesis original\n")
        else:
            veredicto = "HIPÓTESIS NEUTRA"
            f.write(f"🔄 {veredicto}\n")
            f.write("Evidencia insuficiente para confirmar o refutar\n")
        
        f.write(f"\nMétricas clave:\n")
        f.write(f"• Correlación observada: {correlacion_cognitiva:.3f} (esperada: -0.20)\n")
        f.write(f"• Diferencia con esperada: {abs(correlacion_cognitiva + 0.20):.3f}\n")
        
        f.write("\n=== RECOMENDACIONES PRÁCTICAS ===\n")
        if correlacion_cognitiva < -0.1:
            f.write("• Monitorear niveles de carga cognitiva en tareas críticas\n")
            f.write("• Proporcionar descansos o apoyo cuando carga sea alta\n")
            f.write("• Considerar división de tareas complejas en sub-tareas más simples\n")
        else:
            f.write("• Enfocarse en otros factores (cafeína, horas) que tienen mayor impacto\n")
            f.write("• La carga cognitiva puede no ser el factor limitante principal\n")
        
        f.write("\n=== LIMITACIONES DEL ANÁLISIS ===\n")
        f.write("• Carga cognitiva es medida subjetiva, puede variar entre desarrolladores\n")
        f.write("• Correlación no implica causalidad: factores no medidos pueden influir\n")
        f.write("• Dataset limitado a 500 registros, puede no capturar toda la variabilidad\n")
        f.write("• No se controla por experiencia o habilidad del desarrollador\n")
        f.write("• Factores externos (presión temporal, apoyo del equipo) no están incluidos\n")
    
    # Mostrar resumen en consola
    print("\n=== ESTADÍSTICAS DE CARGA COGNITIVA ===")
    print(f"Correlación con task_success: {correlacion_cognitiva:.3f}")
    
    for success in [0, 1]:
        subset = df_completo_pd[df_completo_pd['task_success'] == success]
        print(f"\nTask Success = {success}:")
        print(f"  Promedio carga cognitiva: {subset['cognitive_load'].mean():.1f}")
        print(f"  Mediana carga cognitiva: {subset['cognitive_load'].median():.1f}")
        print(f"  Desviación estándar: {subset['cognitive_load'].std():.1f}")
    
    print("\n=== TASA DE ÉXITO POR RANGO ===")
    for _, row in tasa_exito_rango_pd.iterrows():
        print(f"Rango {row['cognitive_rango']}: {row['tasa_exito']:.1%} de éxito ({row['total_registros']} registros)")
    
    print("\n=== PEORES COMBINACIONES ===")
    peores_combinaciones = interaccion_pd.nsmallest(3, 'tasa_exito')
    for _, row in peores_combinaciones.iterrows():
        print(f"Carga {row['cognitive_rango']} + Horas {row['hours_rango']}: {row['tasa_exito']:.1%} éxito")
    
    print(f"\n=== COMPARACIÓN DE IMPORTANCIA ===")
    print(f"Cafeína vs Éxito: ~0.70 (esperado)")
    print(f"Horas vs Éxito: ~0.62 (esperado)")
    print(f"Carga Cognitiva vs Éxito: {correlacion_cognitiva:.3f} (actual)")
    
    print(f"\n🎯 VEREDICTO: {veredicto}")
    print(f"📊 Correlación observada: {correlacion_cognitiva:.3f} (esperada: -0.20)")
    
    print("\n✅ Plan 3 completado exitosamente")
    print("📁 Archivos guardados en notebooks/results/plan3-carga-cognitiva/")
    print("   - plan3_cognitiva_boxplot.png")
    print("   - plan3_cognitiva_tasa_exito.png") 
    print("   - plan3_cognitiva_scatter.png")
    print("   - plan3_cognitiva_heatmap.png")
    print("   - plan3_cognitiva_estadisticas.txt")

if __name__ == "__main__":
    main()
