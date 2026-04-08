# Plan 2: Hipótesis Horas de Código - Análisis de Productividad

Evaluar si más horas de programación aumentan la probabilidad de éxito en las tareas, identificando posibles rendimientos decrecientes.

## Hipótesis
"Más horas codificando aumenta la probabilidad de éxito"
### Correlación esperada: +0.62

## Enfoque Híbrido
1. **PySpark** → Procesamiento masivo de datos y cálculos estadísticos
2. **pandas** → Conversión para análisis y visualización
3. **matplotlib/seaborn** → Generación de gráficos informativos

## Paso 1 — Configuración inicial

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, sum as spark_sum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("Hipotesis_Horas_Codigo") \
    .getOrCreate()

# Cargar dataset local
df_spark = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)
```

## Paso 2 — Análisis en PySpark

### 2.1 Promedio de horas por task_success
```python
# Promedio de horas agrupado por éxito
promedio_horas = df_spark.groupBy("task_success") \
    .agg(
        avg("hours_coding").alias("avg_hours"),
        avg("commits").alias("avg_commits"),
        count("*").alias("total_registros")
    )

promedio_horas.show()
```

### 2.2 Calcular correlación
```python
# Correlación entre horas y éxito
correlacion_horas = df_spark.stat.corr("hours_coding", "task_success")
print(f"Correlación horas_coding vs task_success: {correlacion_horas:.3f}")

# Correlación entre commits y éxito
correlacion_commits = df_spark.stat.corr("commits", "task_success")
print(f"Correlación commits vs task_success: {correlacion_commits:.3f}")
```

### 2.3 Crear rangos de horas y calcular productividad
```python
# Crear rangos de horas
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

# Estadísticas por rango
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
```

### 2.4 Análisis de rendimientos decrecientes
```python
# Calcular tasa de éxito incremental por cada hora adicional
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Ordenar por horas y calcular tasa de éxito acumulada
window_spec = Window.orderBy("hours_coding")

horas_exito = df_spark.groupBy("hours_coding") \
    .agg(avg("task_success").alias("tasa_exito"), count("*").alias("count")) \
    .filter(col("count") >= 5) \  # Solo considerar horas con suficientes datos
    .orderBy("hours_coding")

horas_exito.show()
```

## Paso 3 — Convertir a pandas para visualización

```python
# Convertir resultados a pandas
promedio_horas_pd = promedio_horas.toPandas()
estadisticas_rango_pd = estadisticas_rango.toPandas()
horas_exito_pd = horas_exito.toPandas()
df_completo_pd = df_spark.toPandas()

# Detener Spark
spark.stop()
```

## Paso 4 — Visualizaciones

### Gráfico 1: Boxplot de horas por task_success
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_completo_pd, x="task_success", y="hours_coding")
plt.title("Distribución de Horas de Código por Éxito de Tarea")
plt.xlabel("Éxito de Tarea (0=No, 1=Sí)")
plt.ylabel("Horas de Código")
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 2: Histograma de horas coloreado por task_success
```python
plt.figure(figsize=(12, 6))
sns.histplot(data=df_completo_pd, x="hours_coding", hue="task_success", 
             bins=15, alpha=0.7, kde=True)
plt.title("Distribución de Horas de Código por Éxito")
plt.xlabel("Horas de Código")
plt.ylabel("Frecuencia")
plt.legend(title="Éxito", labels=["No", "Sí"])
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 3: Barplot de tasa de éxito por rango de horas
```python
plt.figure(figsize=(12, 6))

# Crear subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Tasa de éxito por rango
sns.barplot(data=estadisticas_rango_pd, x="hours_rango", y="tasa_exito", ax=ax1)
ax1.set_title("Tasa de Éxito por Rango de Horas")
ax1.set_xlabel("Rango de Horas")
ax1.set_ylabel("Tasa de Éxito")
ax1.set_ylim(0, 1)

# Agregar etiquetas
for i, row in estadisticas_rango_pd.iterrows():
    ax1.text(i, row.tasa_exito + 0.01, f"{row.tasa_exito:.2f}", 
             ha='center', va='bottom')

# Productividad por rango
sns.barplot(data=estadisticas_rango_pd, x="hours_rango", y="avg_productivity", ax=ax2)
ax2.set_title("Productividad (commits/hora) por Rango")
ax2.set_xlabel("Rango de Horas")
ax2.set_ylabel("Commits por Hora")

plt.tight_layout()
plt.show()
```

### Gráfico 4: Línea de tendencia de éxito por horas exactas
```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=horas_exito_pd, x="hours_coding", y="tasa_exito", 
             marker='o', markersize=8)
plt.title("Tasa de Éxito por Horas Exactas de Código")
plt.xlabel("Horas de Código")
plt.ylabel("Tasa de Éxito")
plt.grid(True, alpha=0.3)
plt.xlim(0, max(horas_exito_pd['hours_coding']) + 1)
plt.show()
```

## Paso 5 — Análisis de resultados

### Información de valor a extraer:
- **Correlación numérica**: Fuerza de la relación horas-éxito
- **Rango óptimo**: Identificar el sweet spot de horas
- **Rendimientos decrecientes**: Punto donde más horas no ayudan
- **Productividad**: Eficiencia (commits/hora) por rango

### Código para análisis final:
```python
print("=== ESTADÍSTICAS DE HORAS DE CÓDIGO ===")
print(f"Correlación con task_success: {correlacion_horas:.3f}")
print(f"Correlación commits con task_success: {correlacion_commits:.3f}")

for success in [0, 1]:
    subset = df_completo_pd[df_completo_pd['task_success'] == success]
    print(f"\nTask Success = {success}:")
    print(f"  Promedio horas: {subset['hours_coding'].mean():.1f} h")
    print(f"  Mediana horas: {subset['hours_coding'].median():.1f} h")
    print(f"  Promedio commits: {subset['commits'].mean():.1f}")

print("\n=== ANÁLISIS POR RANGO ===")
for _, row in estadisticas_rango_pd.iterrows():
    print(f"Rango '{row['hours_rango']}':")
    print(f"  Tasa éxito: {row['tasa_exito']:.1%}")
    print(f"  Horas promedio: {row['avg_hours']:.1f} h")
    print(f"  Productividad: {row['avg_productivity']:.2f} commits/hora")

# Identificar rango óptimo
rango_optimo = estadisticas_rango_pd.loc[estadisticas_rango_pd['tasa_exito'].idxmax()]
print(f"\n=== RANGO ÓPTIMO ===")
print(f"Mejor rango: '{rango_optimo['hours_rango']}' con {rango_optimo['tasa_exito']:.1%} de éxito")
```

## Veredicto esperado
- ✅ **CONFIRMADA** si correlación > 0.5 y hay un rango óptimo claro
- ❌ **REFUTADA** si correlación < 0.3 o no hay patrón claro
- 🔄 **PARCIAL** si hay correlación pero con rendimientos decrecientes

## ✅ Criterio de completado
1. Correlación calculada y mostrada con interpretación
2. Tres gráficos generados con títulos descriptivos y anotaciones
3. Estadísticas comparativas completas (promedio, mediana, min/max, desviación)
4. Análisis de rendimientos decrecientes si aplica
5. Veredicto claro con sustento numérico
6. **Cumplimiento del estándar de calidad**: Documentación detallada con explicaciones comprensibles

## 📋 Estándar de Calidad Obligatorio

**REQUISITO**: Este plan debe seguir el estándar de documentación detallada:

### Archivo de Estadísticas (`plan2_horas_estadisticas.txt`):
- ✅ Metodología clara (variables, técnicas, herramientas)
- ✅ Definición de rangos de horas con equivalencias prácticas
- ✅ Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
- ✅ Análisis por categorías con interpretación
- ✅ Insights clave: rendimientos decrecientes, punto óptimo
- ✅ Veredicto con contexto y métricas
- ✅ Recomendaciones prácticas basadas en evidencia
- ✅ Limitaciones del análisis documentadas

### Gráficos (3 requeridos):
- ✅ Títulos descriptivos con hipótesis
- ✅ Etiquetas claras en ejes
- ✅ Anotaciones estadísticas (medianas, porcentajes)
- ✅ Referencias visuales para rangos de horas
- ✅ Equivalencias prácticas (ej: horas laborales)

### Formato:
- Script: `plan2_horas_codigo_analysis.py`
- Estadísticas: `plan2_horas_estadisticas.txt`
- Gráficos: `plan2_horas_boxplot.png`, `plan2_horas_histograma.png`, `plan2_horas_tasa_exito.png`
- Carpeta: `notebooks/results/plan2-horas-codigo/`

**Sin excepción**: Plan no está "completado" hasta cumplir todos estos estándares.

## Archivos de salida sugeridos
- `notebooks/results/plan2-horas-codigo/plan2_horas_boxplot.png`
- `notebooks/results/plan2-horas-codigo/plan2_horas_histograma.png`
- `notebooks/results/plan2-horas-codigo/plan2_horas_tasa_exito.png`
- `notebooks/results/plan2-horas-codigo/plan2_horas_estadisticas.txt`
