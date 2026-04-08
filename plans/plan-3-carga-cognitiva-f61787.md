# Plan 3: Hipótesis Carga Cognitiva - Impacto en el Éxito

Analizar si la alta carga cognitiva reduce el éxito en las tareas y cómo interactúa con las horas de trabajo.

## Hipótesis
"Alta carga cognitiva reduce el task_success"
### Correlación esperada: -0.20 (débil negativa)

## Enfoque Híbrido
1. **PySpark** → Procesamiento y análisis de correlaciones cruzadas
2. **pandas** → Conversión para visualizaciones complejas
3. **matplotlib/seaborn** → Gráficos de interacción y distribución

## Paso 1 — Configuración inicial

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, corr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("Hipotesis_Carga_Cognitiva") \
    .getOrCreate()

# Cargar dataset local
df_spark = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)
```

## Paso 2 — Análisis en PySpark

### 2.1 Promedio de carga cognitiva por task_success
```python
# Promedio de carga cognitiva agrupado por éxito
promedio_cognitiva = df_spark.groupBy("task_success") \
    .agg(
        avg("cognitive_load").alias("avg_cognitive_load"),
        avg("hours_coding").alias("avg_hours"),
        count("*").alias("total_registros")
    )

promedio_cognitiva.show()
```

### 2.2 Calcular correlaciones principales
```python
# Correlación carga cognitiva vs éxito
correlacion_cognitiva = df_spark.stat.corr("cognitive_load", "task_success")
print(f"Correlación cognitive_load vs task_success: {correlacion_cognitiva:.3f}")

# Correlación carga cognitiva vs horas
correlacion_horas_cognitiva = df_spark.stat.corr("cognitive_load", "hours_coding")
print(f"Correlación cognitive_load vs hours_coding: {correlacion_horas_cognitiva:.3f}")

# Correlación carga cognitiva vs distracciones
correlacion_distracciones = df_spark.stat.corr("cognitive_load", "distractions")
print(f"Correlación cognitive_load vs distractions: {correlacion_distracciones:.3f}")
```

### 2.3 Crear rangos de carga cognitiva
```python
# Crear rangos de carga cognitiva
df_con_rangos = df_spark.withColumn(
    "cognitive_rango",
    when(col("cognitive_load") <= 3, "baja")
    .when((col("cognitive_load") >= 4) & (col("cognitive_load") <= 6), "media")
    .otherwise("alta")
)

# Calcular tasa de éxito por rango cognitivo
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
```

### 2.4 Análisis de interacción: carga cognitiva + horas
```python
# Crear combinación de rangos
df_con_interaccion = df_con_rangos.withColumn(
    "hours_rango",
    when(col("hours_coding") < 3, "pocas")
    .when((col("hours_coding") >= 3) & (col("hours_coding") < 6), "moderadas")
    .when((col("hours_coding") >= 6) & (col("hours_coding") < 9), "muchas")
    .otherwise("excesivas")
)

# Análisis de combinación carga cognitiva + horas
interaccion = df_con_interaccion.groupBy("cognitive_rango", "hours_rango") \
    .agg(
        avg("task_success").alias("tasa_exito"),
        count("*").alias("total_registros")
    ) \
    .filter(col("total_registros") >= 5) \  # Solo combinaciones con datos suficientes
    .orderBy("cognitive_rango", "hours_rango")

interaccion.show()
```

### 2.5 Análisis de factores que influyen en la carga cognitiva
```python
# Qué factores se correlacionan con alta carga cognitiva
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
```

## Paso 3 — Convertir a pandas para visualización

```python
# Convertir resultados a pandas
promedio_cognitiva_pd = promedio_cognitiva.toPandas()
tasa_exito_rango_pd = tasa_exito_rango.toPandas()
interaccion_pd = interaccion.toPandas()
df_completo_pd = df_spark.toPandas()

# Detener Spark
spark.stop()
```

## Paso 4 — Visualizaciones

### Gráfico 1: Boxplot de carga cognitiva por task_success
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_completo_pd, x="task_success", y="cognitive_load")
plt.title("Distribución de Carga Cognitiva por Éxito de Tarea")
plt.xlabel("Éxito de Tarea (0=No, 1=Sí)")
plt.ylabel("Carga Cognitiva (escala 1-10)")
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 2: Barplot de tasa de éxito por rango cognitivo
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=tasa_exito_rango_pd, x="cognitive_rango", y="tasa_exito")
plt.title("Tasa de Éxito por Rango de Carga Cognitiva")
plt.xlabel("Rango de Carga Cognitiva")
plt.ylabel("Tasa de Éxito")
plt.ylim(0, 1)

# Agregar etiquetas de valores
for i, row in tasa_exito_rango_pd.iterrows():
    plt.text(i, row.tasa_exito + 0.01, f"{row.tasa_exito:.2f}", 
             ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 3: Scatter plot de carga cognitiva vs horas coloreado por task_success
```python
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_completo_pd, x="cognitive_load", y="hours_coding", 
                hue="task_success", size="commits", alpha=0.7, sizes=(20, 200))
plt.title("Interacción: Carga Cognitiva vs Horas de Código")
plt.xlabel("Carga Cognitiva (escala 1-10)")
plt.ylabel("Horas de Código")
plt.legend(title="Éxito", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 4: Heatmap de interacción carga cognitiva + horas
```python
# Pivot para heatmap
heatmap_data = interaccion_pd.pivot(index="cognitive_rango", 
                                  columns="hours_rango", 
                                  values="tasa_exito")

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", 
           vmin=0, vmax=1, cbar_kws={'label': 'Tasa de Éxito'})
plt.title("Tasa de Éxito: Carga Cognitiva vs Horas de Código")
plt.xlabel("Rango de Horas")
plt.ylabel("Rango de Carga Cognitiva")
plt.tight_layout()
plt.show()
```

## Paso 5 — Análisis de resultados

### Información de valor a extraer:
- **Correlación débil**: Confirmar si la correlación es realmente baja
- **Factores influyentes**: Qué aumenta la carga cognitiva
- **Interacción crítica**: Carga alta + muchas horas = desastre
- **Comparación relativa**: Importancia vs otros factores

### Código para análisis final:
```python
print("=== ESTADÍSTICAS DE CARGA COGNITIVA ===")
print(f"Correlación con task_success: {correlacion_cognitiva:.3f}")

for success in [0, 1]:
    subset = df_completo_pd[df_completo_pd['task_success'] == success]
    print(f"\nTask Success = {success}:")
    print(f"  Promedio carga cognitiva: {subset['cognitive_load'].mean():.1f}")
    print(f"  Mediana carga cognitiva: {subset['cognitive_load'].median():.1f}")

print("\n=== ANÁLISIS POR RANGO COGNITIVO ===")
for _, row in tasa_exito_rango_pd.iterrows():
    print(f"Rango '{row['cognitive_rango']}':")
    print(f"  Tasa éxito: {row['tasa_exito']:.1%}")
    print(f"  Horas promedio: {row['avg_hours']:.1f} h")
    print(f"  Distracciones promedio: {row['avg_distractions']:.1f}")

print("\n=== PEORES COMBINACIONES ===")
peores_combinaciones = interaccion_pd.nsmallest(3, 'tasa_exito')
for _, row in peores_combinaciones.iterrows():
    print(f"Carga {row['cognitive_rango']} + Horas {row['hours_rango']}: "
          f"{row['tasa_exito']:.1%} éxito")

# Comparación con otras correlaciones
print(f"\n=== COMPARACIÓN DE IMPORTANCIA ===")
print(f"Cafeína vs Éxito: ~0.70 (esperado)")
print(f"Horas vs Éxito: ~0.62 (esperado)")
print(f"Carga Cognitiva vs Éxito: {correlacion_cognitiva:.3f} (actual)")
```

## Veredicto esperado
- ✅ **CONFIRMADA (débil)** si correlación entre -0.1 y -0.3
- ❌ **REFUTADA** si correlación cercana a 0 o positiva
- 🔄 **CONTEXTUAL** si la correlación es débil pero la interacción con horas es significativa

## ✅ Criterio de completado
1. Correlación calculada y contextualizada
2. Cuatro gráficos generados (incluyendo heatmap de interacción)
3. Análisis de factores que influyen en la carga cognitiva
4. Comparación con importancia de otras variables

## Archivos de salida sugeridos
- `results/plan3_cognitiva_boxplot.png`
- `results/plan3_cognitiva_rangos.png`
- `results/plan3_cognitiva_scatter.png`
- `results/plan3_cognitiva_heatmap.png`
- `results/plan3_cognitiva_estadisticas.txt`

## 📋 Estándar de Calidad Obligatorio

**REQUISITO**: Este plan debe seguir el estándar de documentación detallada:

### Archivo de Estadísticas (`plan3_cognitiva_estadisticas.txt`):
- ✅ Metodología clara (variables, técnicas, herramientas)
- ✅ Definición de rangos de carga cognitiva con equivalencias prácticas
- ✅ Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
- ✅ Análisis por categorías con interpretación
- ✅ Insights clave: umbral crítico, patrones de rendimiento
- ✅ Veredicto con contexto y métricas
- ✅ Recomendaciones prácticas basadas en evidencia
- ✅ Limitaciones del análisis documentadas

### Gráficos (3 requeridos):
- ✅ Títulos descriptivos con hipótesis
- ✅ Etiquetas claras en ejes
- ✅ Anotaciones estadísticas (medianas, porcentajes)
- ✅ Referencias visuales para rangos de carga
- ✅ Equivalencias prácticas (ej: niveles de estrés)

### Formato:
- Script: `plan3_cognitiva_analysis.py`
- Estadísticas: `plan3_cognitiva_estadisticas.txt`
- Gráficos: `plan3_cognitiva_boxplot.png`, `plan3_cognitiva_histograma.png`, `plan3_cognitiva_tasa_exito.png`
- Carpeta: `notebooks/results/plan3-carga-cognitiva/`

**Sin excepción**: Plan no está "completado" hasta cumplir todos estos estándares.

## Archivos de salida sugeridos
- `notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_boxplot.png`
- `notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_histograma.png`
- `notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_tasa_exito.png`
- `notebooks/results/plan3-carga-cognitiva/plan3_cognitiva_estadisticas.txt`
- `results/plan3_cognitiva_estadisticas.txt`
