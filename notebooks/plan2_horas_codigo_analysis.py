#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan 2: Hipótesis Horas de Código - Análisis de Productividad
Evaluar si más horas de programación aumentan la probabilidad de éxito en las tareas,
identificando posibles rendimientos decrecientes.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, sum as spark_sum
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("🧪 Iniciando Plan 2: Análisis de Hipótesis Horas de Código")
    
    # Paso 1 — Configuración inicial
    print("\n📊 Paso 1: Configuración inicial")
    
    spark = SparkSession.builder \
        .appName("Hipotesis_Horas_Codigo") \
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
    
    # 2.1 Promedio de horas por task_success
    print("🔍 2.1 Promedio de horas por task_success")
    promedio_horas = df_spark.groupBy("task_success") \
        .agg(
            avg("hours_coding").alias("avg_hours"),
            avg("commits").alias("avg_commits"),
            count("*").alias("total_registros")
        )
    
    promedio_horas.show()
    
    # 2.2 Calcular correlación
    print("🔍 2.2 Correlación entre horas y éxito")
    correlacion_horas = df_spark.stat.corr("hours_coding", "task_success")
    print(f"Correlación horas_coding vs task_success: {correlacion_horas:.3f}")
    
    correlacion_commits = df_spark.stat.corr("commits", "task_success")
    print(f"Correlación commits vs task_success: {correlacion_commits:.3f}")
    
    # 2.3 Crear rangos de horas y calcular productividad
    print("🔍 2.3 Crear rangos de horas")
    df_con_rangos = df_spark.withColumn(
        "hours_rango",
        when(col("hours_coding") < 3, "pocas")
        .when((col("hours_coding") >= 3) & (col("hours_coding") < 6), "moderadas")
        .when((col("hours_coding") >= 6) & (col("hours_coding") < 9), "muchas")
        .otherwise("excesivas")
    )
    
    # Calcular productividad ratio (commits/hora)
    df_con_productividad = df_con_rangos.withColumn(
        "productivity_ratio",
        when(col("hours_coding") > 0, col("commits") / col("hours_coding")).otherwise(0)
    )
    
    estadisticas_rango = df_con_productividad.groupBy("hours_rango") \
        .agg(
            avg("task_success").alias("tasa_exito"),
            avg("hours_coding").alias("avg_hours"),
            avg("commits").alias("avg_commits"),
            avg("productivity_ratio").alias("avg_productivity"),
            count("*").alias("total_registros")
        ) \
        .orderBy("avg_hours")
    
    estadisticas_rango.show()
    
    # 2.4 Análisis de rendimientos decrecientes
    print("🔍 2.4 Análisis de rendimientos decrecientes")
    horas_exito = df_spark.groupBy("hours_coding") \
        .agg(avg("task_success").alias("tasa_exito"), count("*").alias("count")) \
        .filter(col("count") >= 5) \
        .orderBy("hours_coding")
    
    horas_exito.show()
    
    # Paso 3 — Convertir a pandas para visualización
    print("\n🔄 Paso 3: Convertir a pandas para visualización")
    
    promedio_horas_pd = promedio_horas.toPandas()
    estadisticas_rango_pd = estadisticas_rango.toPandas()
    horas_exito_pd = horas_exito.toPandas()
    df_completo_pd = df_spark.toPandas()
    
    spark.stop()
    print("✅ Sesión Spark detenida")
    
    # Paso 4 — Visualizaciones
    print("\n📊 Paso 4: Generando visualizaciones")
    
    # Crear directorio de resultados
    os.makedirs("notebooks/results/plan2-horas-codigo", exist_ok=True)
    
    # Gráfico 1: Boxplot de horas por task_success
    print("📈 Gráfico 1: Boxplot de horas por task_success")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_completo_pd, x="task_success", y="hours_coding")
    plt.title("Hipótesis Horas de Código: Distribución de Horas por Éxito de Tarea", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Éxito de Tarea (0=Fracaso, 1=Éxito)", fontsize=12)
    plt.ylabel("Horas de Código", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Agregar medianas
    for i, success in enumerate([0, 1]):
        subset = df_completo_pd[df_completo_pd['task_success'] == success]
        median = subset['hours_coding'].median()
        plt.text(i, median + 0.2, f"Mediana: {median:.1f}h", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan2-horas-codigo/plan2_horas_boxplot.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 2: Histograma de horas coloreado por task_success
    print("📈 Gráfico 2: Histograma de horas por task_success")
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df_completo_pd, x="hours_coding", hue="task_success", 
                 bins=15, alpha=0.7, kde=True)
    plt.title("Hipótesis Horas de Código: Distribución de Horas por Éxito", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Horas de Código", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.legend(title="Éxito", labels=["Éxito", "Fracaso"])
    plt.grid(True, alpha=0.3)
    
    # Agregar líneas de referencia para rangos
    plt.axvline(x=3, color='orange', linestyle='--', alpha=0.7, label='Límite < 3h')
    plt.axvline(x=6, color='red', linestyle='--', alpha=0.7, label='Límite < 6h')
    plt.axvline(x=9, color='purple', linestyle='--', alpha=0.7, label='Límite < 9h')
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan2-horas-codigo/plan2_horas_histograma.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 3: Tasa de éxito por rango de horas
    print("📈 Gráfico 3: Tasa de éxito por rango de horas")
    plt.figure(figsize=(14, 8))
    
    # Tasa de éxito por rango
    ax = sns.barplot(data=estadisticas_rango_pd, x="hours_rango", y="tasa_exito")
    plt.title("Hipótesis Horas de Código: Tasa de Éxito por Rango de Horas", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Rango de Horas de Código", fontsize=12)
    plt.ylabel("Tasa de Éxito", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Agregar etiquetas de porcentaje y número de desarrolladores
    for i, row in estadisticas_rango_pd.iterrows():
        ax.text(i, row.tasa_exito + 0.03, f"{row.tasa_exito:.1%}\n({row.total_registros} devs)", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Agregar descripción de rangos en el eje X
    rango_labels = {
        'pocas': '< 3h\n(Medio día)',
        'moderadas': '3-6h\n(3/4 día)',
        'muchas': '6-9h\n(Día completo)',
        'excesivas': '> 9h\n(Más de un día)'
    }
    
    # Obtener etiquetas actuales y reemplazar
    ax = plt.gca()
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    new_labels = [rango_labels.get(label, label) for label in current_labels]
    ax.set_xticks(range(len(current_labels)))
    ax.set_xticklabels(new_labels)
    
    plt.tight_layout()
    plt.savefig("notebooks/results/plan2-horas-codigo/plan2_horas_tasa_exito.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Paso 5 — Análisis de resultados
    print("\n📋 Paso 5: Análisis de resultados")
    
    # Guardar estadísticas en archivo
    with open("notebooks/results/plan2-horas-codigo/plan2_horas_estadisticas.txt", "w", encoding='utf-8') as f:
        f.write("=== PLAN 2: ANÁLISIS DE HIPÓTESIS HORAS DE CÓDIGO ===\n")
        f.write("Hipótesis: 'Más horas codificando aumenta la probabilidad de éxito'\n")
        f.write("Correlación esperada: +0.62\n\n")
        
        f.write("=== METODOLOGÍA ===\n")
        f.write("• Dataset: 500 registros de productividad de desarrolladores\n")
        f.write("• Variable independiente: hours_coding (horas de programación por sesión)\n")
        f.write("• Variable dependiente: task_success (0=fallo, 1=éxito)\n")
        f.write("• Análisis: Correlación de Pearson + agrupación por rangos + análisis de productividad\n")
        f.write("• Herramientas: PySpark para procesamiento, pandas/matplotlib para visualización\n\n")
        
        f.write("=== DEFINICIÓN DE RANGOS DE HORAS ===\n")
        f.write("• Rango POCAS: < 3 horas (equivalente a medio día laboral)\n")
        f.write("• Rango MODERADAS: 3-6 horas (equivalente a 3/4 día laboral)\n")
        f.write("• Rango MUCHAS: 6-9 horas (equivalente a día laboral completo)\n")
        f.write("• Rango EXCESIVAS: > 9 horas (equivalente a más de un día laboral)\n")
        f.write("• Referencia: Jornada laboral estándar de 8 horas\n\n")
        
        f.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        f.write(f"Correlación de Pearson: {correlacion_horas:.3f}\n")
        f.write("Interpretación de correlación:\n")
        f.write("• 0.0-0.3: Correlación débil\n")
        f.write("• 0.3-0.5: Correlación moderada\n")
        f.write("• 0.5-0.7: Correlación fuerte\n")
        f.write("• 0.7-1.0: Correlación muy fuerte\n\n")
        
        f.write("Análisis por grupo de éxito:\n\n")
        
        for success in [0, 1]:
            subset = df_completo_pd[df_completo_pd['task_success'] == success]
            grupo = "FRACASO" if success == 0 else "ÉXITO"
            f.write(f"Grupo {grupo} (task_success = {success}):\n")
            f.write(f"  • Promedio horas: {subset['hours_coding'].mean():.1f} h\n")
            f.write(f"  • Mediana horas: {subset['hours_coding'].median():.1f} h\n")
            f.write(f"  • Desviación estándar: {subset['hours_coding'].std():.1f} h\n")
            f.write(f"  • Mínimo: {subset['hours_coding'].min():.1f} h\n")
            f.write(f"  • Máximo: {subset['hours_coding'].max():.1f} h\n")
            f.write(f"  • Promedio commits: {subset['commits'].mean():.1f}\n")
            f.write(f"  • Total registros: {len(subset)}\n\n")
        
        # Calcular diferencia clave
        grupo_exito = df_completo_pd[df_completo_pd['task_success'] == 1]
        grupo_fracaso = df_completo_pd[df_completo_pd['task_success'] == 0]
        diff_horas = grupo_exito['hours_coding'].mean() - grupo_fracaso['hours_coding'].mean()
        diff_pct = (diff_horas / grupo_fracaso['hours_coding'].mean()) * 100
        
        f.write("Diferencia clave:\n")
        f.write(f"  • Los desarrolladores con éxito trabajan {diff_horas:.1f} horas más en promedio\n")
        f.write(f"  • Esto representa un {diff_pct:.1f}% más de tiempo de código\n\n")
        
        f.write("=== ANÁLISIS POR RANGOS DE HORAS ===\n")
        f.write("Tasa de éxito y productividad por rango:\n\n")
        
        for _, row in estadisticas_rango_pd.iterrows():
            f.write(f"Rango {row['hours_rango'].upper()}:\n")
            f.write(f"  • Tasa de éxito: {row['tasa_exito']:.1%} ({row['tasa_exito']:.3f})\n")
            f.write(f"  • Horas promedio: {row['avg_hours']:.1f} h\n")
            f.write(f"  • Commits promedio: {row['avg_commits']:.1f}\n")
            f.write(f"  • Productividad: {row['avg_productivity']:.2f} commits/hora\n")
            f.write(f"  • Total registros: {row['total_registros']}\n")
            f.write(f"  • Proporción del dataset: {(row['total_registros']/len(df_completo_pd)*100):.1f}%\n")
            
            # Interpretación
            if row['hours_rango'] == 'pocas':
                interpretacion = "Tiempo insuficiente para completar tareas complejas"
            elif row['hours_rango'] == 'moderadas':
                interpretacion = "Tiempo adecuado para tareas moderadamente complejas"
            elif row['hours_rango'] == 'muchas':
                interpretacion = "Tiempo suficiente para tareas complejas"
            else:  # excesivas
                interpretacion = "Posible fatiga o sobreesfuerzo afectando rendimiento"
            
            f.write(f"  • Interpretación: {interpretacion}\n\n")
        
        f.write("=== INSIGHTS CLAVE ===\n")
        f.write("1. RELACIÓN HORAS-ÉXITO:\n")
        if correlacion_horas > 0.5:
            f.write(f"   • Correlación fuerte ({correlacion_horas:.3f}) indica relación positiva clara\n")
        elif correlacion_horas > 0.3:
            f.write(f"   • Correlación moderada ({correlacion_horas:.3f}) indica relación parcial\n")
        else:
            f.write(f"   • Correlación débil ({correlacion_horas:.3f}) indica relación limitada\n")
        
        f.write("2. RENDIMIENTOS DECRECIENTES:\n")
        rango_optimo = estadisticas_rango_pd.loc[estadisticas_rango_pd['tasa_exito'].idxmax()]
        f.write(f"   • Rango óptimo: {rango_optimo['hours_rango']} con {rango_optimo['tasa_exito']:.1%} éxito\n")
        
        # Verificar si hay rendimientos decrecientes
        tasas_por_rango = dict(zip(estadisticas_rango_pd['hours_rango'], estadisticas_rango_pd['tasa_exito']))
        orden_rangos = ['pocas', 'moderadas', 'muchas', 'excesivas']
        tasas_ordenadas = [tasas_por_rango.get(r, 0) for r in orden_rangos if r in tasas_por_rango]
        
        if len(tasas_ordenadas) >= 3:
            if tasas_ordenadas[-1] < tasas_ordenadas[-2]:
                f.write("   • Evidencia de rendimientos decrecientes en rango excesivo\n")
            else:
                f.write("   • Sin evidencia clara de rendimientos decrecientes\n")
        
        f.write("3. PRODUCTIVIDAD:\n")
        prod_max = estadisticas_rango_pd.loc[estadisticas_rango_pd['avg_productivity'].idxmax()]
        f.write(f"   • Máxima productividad: {prod_max['hours_rango']} ({prod_max['avg_productivity']:.2f} commits/hora)\n")
        
        f.write("4. UMPRAL MÍNIMO:\n")
        rango_min = estadisticas_rango_pd[estadisticas_rango_pd['avg_hours'] == estadisticas_rango_pd['avg_hours'].min()].iloc[0]
        if rango_min['tasa_exito'] < 0.2:
            f.write(f"   • Rango '{rango_min['hours_rango']}' muestra baja tasa de éxito ({rango_min['tasa_exito']:.1%})\n")
            f.write("   • Sugiere umbral mínimo de horas necesario para éxito\n")
        
        f.write("\n=== VEREDICTO FINAL ===\n")
        if correlacion_horas >= 0.5:
            veredicto = "HIPÓTESIS CONFIRMADA"
            f.write(f"✅ {veredicto}\n")
            f.write("Evidencia fuerte que apoya la hipótesis original\n")
        elif correlacion_horas >= 0.3:
            veredicto = "HIPÓTESIS PARCIALMENTE CONFIRMADA"
            f.write(f"🔄 {veredicto}\n")
            f.write("Evidencia moderada con posibles factores adicionales\n")
        else:
            veredicto = "HIPÓTESIS REFUTADA"
            f.write(f"❌ {veredicto}\n")
            f.write("Evidencia insuficiente para apoyar la hipótesis\n")
        
        f.write(f"\nMétricas clave:\n")
        f.write(f"• Correlación observada: {correlacion_horas:.3f} (esperada: +0.62)\n")
        f.write(f"• Precisión: {abs(correlacion_horas - 0.62):.3f} de diferencia con esperada\n")
        f.write(f"• Rango óptimo: {rango_optimo['hours_rango']} ({rango_optimo['tasa_exito']:.1%} éxito)\n")
        
        f.write("\n=== RECOMENDACIONES PRÁCTICAS ===\n")
        if correlacion_horas > 0.3:
            f.write("• Asignar suficiente tiempo para tareas críticas (rango óptimo identificado)\n")
            if rango_optimo['hours_rango'] == 'muchas':
                f.write("• Planificar jornadas completas para tareas complejas\n")
            elif rango_optimo['hours_rango'] == 'moderadas':
                f.write("• Optimizar sesiones de 3-6 horas para máximo rendimiento\n")
            
            if 'excesivas' in estadisticas_rango_pd['hours_rango'].values:
                tasa_excesiva = estadisticas_rango_pd[estadisticas_rango_pd['hours_rango'] == 'excesivas']['tasa_exito'].iloc[0]
                if tasa_excesiva < rango_optimo['tasa_exito'] * 0.8:
                    f.write("• Evitar jornadas excesivas (>9h) que pueden reducir rendimiento\n")
        else:
            f.write("• Enfocarse en calidad sobre cantidad de horas\n")
            f.write("• Considerar otros factores además del tiempo de código\n")
        
        f.write("\n=== LIMITACIONES DEL ANÁLISIS ===\n")
        f.write("• Correlación no implica causalidad: otros factores pueden influir\n")
        f.write("• Dataset limitado a 500 registros, puede no representar todas las situaciones\n")
        f.write("• No se controla por dificultad de tareas o experiencia del desarrollador\n")
        f.write("• Factores externos (interrupciones, herramientas) no están incluidos\n")
        f.write("• La productividad medida en commits puede no reflejar calidad del código\n")
    
    # Mostrar resumen en consola
    print("\n=== ESTADÍSTICAS DE HORAS DE CÓDIGO ===")
    print(f"Correlación con task_success: {correlacion_horas:.3f}")
    print(f"Correlación commits con task_success: {correlacion_commits:.3f}")
    
    for success in [0, 1]:
        subset = df_completo_pd[df_completo_pd['task_success'] == success]
        print(f"\nTask Success = {success}:")
        print(f"  Promedio horas: {subset['hours_coding'].mean():.1f} h")
        print(f"  Mediana horas: {subset['hours_coding'].median():.1f} h")
        print(f"  Desviación estándar: {subset['hours_coding'].std():.1f} h")
        print(f"  Promedio commits: {subset['commits'].mean():.1f}")
    
    print("\n=== TASA DE ÉXITO POR RANGO ===")
    for _, row in estadisticas_rango_pd.iterrows():
        print(f"Rango {row['hours_rango']}: {row['tasa_exito']:.1%} de éxito ({row['total_registros']} registros)")
    
    rango_optimo = estadisticas_rango_pd.loc[estadisticas_rango_pd['tasa_exito'].idxmax()]
    print(f"\n🎯 VEREDICTO: {veredicto}")
    print(f"📊 Correlación observada: {correlacion_horas:.3f} (esperada: +0.62)")
    print(f"🎯 Rango óptimo: '{rango_optimo['hours_rango']}' con {rango_optimo['tasa_exito']:.1%} de éxito")
    
    print("\n✅ Plan 2 completado exitosamente")
    print("📁 Archivos guardados en notebooks/results/plan2-horas-codigo/")
    print("   - plan2_horas_boxplot.png")
    print("   - plan2_horas_histograma.png") 
    print("   - plan2_horas_tasa_exito.png")
    print("   - plan2_horas_estadisticas.txt")

if __name__ == "__main__":
    main()
