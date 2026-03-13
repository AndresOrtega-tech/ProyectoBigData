# Plan 4: Hipótesis Bugs Reportados - Calidad vs Éxito

Analizar si reportar más bugs indica menor probabilidad de éxito y explorar la relación entre cantidad y calidad del código.

## Hipótesis
"Más bugs reportados indica menor probabilidad de éxito"
### Correlación esperada: -0.18 (muy débil negativa)

## Enfoque Híbrido
1. **PySpark** → Análisis de distribución y correlaciones
2. **pandas** → Visualización de patrones específicos
3. **matplotlib/seaborn** → Gráficos de distribución y relaciones

## Paso 1 — Configuración inicial

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, sum as spark_sum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("Hipotesis_Bugs_Reportados") \
    .getOrCreate()

# Cargar dataset local
df_spark = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)
```

## Paso 2 — Análisis en PySpark

### 2.1 Estadísticas básicas de bugs
```python
# Distribución de bugs
distribucion_bugs = df_spark.groupBy("bugs_reported") \
    .agg(
        count("*").alias("frecuencia"),
        avg("task_success").alias("tasa_exito"),
        avg("commits").alias("avg_commits"),
        avg("hours_coding").alias("avg_hours")
    ) \
    .orderBy("bugs_reported")

distribucion_bugs.show()
```

### 2.2 Calcular correlaciones
```python
# Correlación bugs vs éxito
correlacion_bugs = df_spark.stat.corr("bugs_reported", "task_success")
print(f"Correlación bugs_reported vs task_success: {correlacion_bugs:.3f}")

# Correlación bugs vs commits (cantidad vs calidad)
correlacion_commits_bugs = df_spark.stat.corr("bugs_reported", "commits")
print(f"Correlación bugs_reported vs commits: {correlacion_commits_bugs:.3f}")

# Correlación bugs vs horas (más tiempo = más bugs?)
correlacion_horas_bugs = df_spark.stat.corr("bugs_reported", "hours_coding")
print(f"Correlación bugs_reported vs hours_coding: {correlacion_horas_bugs:.3f}")
```

### 2.3 Análisis de la mediana y distribución
```python
# Calcular mediana y percentiles
from pyspark.sql.functions import expr

# Estadísticas descriptivas de bugs
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

# Porcentaje de registros con 0 bugs
cero_bugs = df_spark.filter(col("bugs_reported") == 0).count()
porcentaje_cero = (cero_bugs / stats_bugs['total_registros']) * 100
print(f"Registros con 0 bugs: {cero_bugs} ({porcentaje_cero:.1f}%)")
```

### 2.4 Análisis por categorías de bugs
```python
# Crear categorías de bugs
df_con_categorias = df_spark.withColumn(
    "bugs_categoria",
    when(col("bugs_reported") == 0, "cero_bugs")
    .when(col("bugs_reported") == 1, "un_bug")
    .when(col("bugs_reported") == 2, "dos_bugs")
    .otherwise("tres_o_mas")
)

# Análisis por categoría
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
```

### 2.5 Análisis de productividad con bugs
```python
# Calcular commits por bug (solo para registros con bugs > 0)
df_con_productividad = df_spark.filter(col("bugs_reported") > 0) \
    .withColumn("commits_por_bug", col("commits") / col("bugs_reported"))

# Estadísticas de productividad con bugs
productividad_bugs = df_con_productividad.agg(
    avg("commits_por_bug").alias("avg_commits_per_bug"),
    avg("task_success").alias("tasa_exito_con_bugs"),
    count("*").alias("registros_con_bugs")
).collect()[0]

print("=== PRODUCTIVIDAD CON BUGS ===")
print(f"Commits por bug (promedio): {productividad_bugs['avg_commits_per_bug']:.2f}")
print(f"Tasa éxito con bugs: {productividad_bugs['tasa_exito_con_bugs']:.1%}")
print(f"Registros con bugs: {productividad_bugs['registros_con_bugs']}")
```

## Paso 3 — Convertir a pandas para visualización

```python
# Convertir resultados a pandas
distribucion_bugs_pd = distribucion_bugs.toPandas()
categoria_stats_pd = categoria_stats.toPandas()
df_completo_pd = df_spark.toPandas()

# Detener Spark
spark.stop()
```

## Paso 4 — Visualizaciones

### Gráfico 1: Barplot de conteo de bugs por task_success
```python
plt.figure(figsize=(12, 6))
sns.countplot(data=df_completo_pd, x="bugs_reported", hue="task_success")
plt.title("Distribución de Bugs Reportados por Éxito de Tarea")
plt.xlabel("Número de Bugs Reportados")
plt.ylabel("Frecuencia")
plt.legend(title="Éxito", labels=["No", "Sí"])
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 2: Barplot de tasa de éxito por número exacto de bugs
```python
plt.figure(figsize=(12, 6))
sns.barplot(data=distribucion_bugs_pd, x="bugs_reported", y="tasa_exito")
plt.title("Tasa de Éxito por Número de Bugs Reportados")
plt.xlabel("Número de Bugs Reportados")
plt.ylabel("Tasa de Éxito")
plt.ylim(0, 1)

# Agregar etiquetas de valores y frecuencias
for i, row in distribucion_bugs_pd.iterrows():
    plt.text(i, row.tasa_exito + 0.01, f"{row.tasa_exito:.2f}\n(n={row['frecuencia']})", 
             ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 3: Scatter plot de commits vs bugs_reported coloreado por task_success
```python
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_completo_pd, x="bugs_reported", y="commits", 
                hue="task_success", size="hours_coding", alpha=0.7, sizes=(20, 200))
plt.title("Relación Cantidad (Commits) vs Calidad (Bugs)")
plt.xlabel("Bugs Reportados")
plt.ylabel("Commits Realizados")
plt.legend(title="Éxito", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 4: Comparación de métricas por categoría de bugs
```python
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Tasa de éxito por categoría
sns.barplot(data=categoria_stats_pd, x="bugs_categoria", y="tasa_exito", ax=axes[0,0])
axes[0,0].set_title("Tasa de Éxito por Categoría de Bugs")
axes[0,0].set_ylabel("Tasa de Éxito")

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

plt.tight_layout()
plt.show()
```

## Paso 5 — Análisis de resultados

### Información de valor a extraer:
- **Distribución especial**: Mediana = 0, mayoría sin bugs
- **Correlación débil**: Confirmar si es realmente insignificante
- **Calidad vs cantidad**: Relación commits vs bugs
- **Patrones de comportamiento**: Qué caracteriza a quienes reportan bugs

### Código para análisis final:
```python
print("=== ESTADÍSTICAS DE BUGS REPORTADOS ===")
print(f"Correlación con task_success: {correlacion_bugs:.3f}")
print(f"Correlación con commits: {correlacion_commits_bugs:.3f}")
print(f"Correlación con horas: {correlacion_horas_bugs:.3f}")

print(f"\nMediana de bugs: {stats_bugs['mediana']}")
print(f"Registros con 0 bugs: {porcentaje_cero:.1f}%")

print("\n=== ANÁLISIS POR NÚMERO EXACTO DE BUGS ===")
for _, row in distribucion_bugs_pd.iterrows():
    print(f"{int(row['bugs_reported'])} bugs: {row['tasa_exito']:.1%} éxito ({row['frecuencia']} registros)")

print("\n=== INSIGHTS CLAVE ===")
if correlacion_bugs > -0.1:
    print("• La correlación es prácticamente nula - los bugs no predicen el éxito")
else:
    print(f"• Existe una correlación débil negativa de {correlacion_bugs:.3f}")

if porcentaje_cero > 50:
    print("• La mayoría de las sesiones no reportan bugs - podría indicar:")
    print("  - Tareas simples sin bugs evidentes")
    print("  - Falta de detección/reporting de bugs")
    print("  - Código de alta calidad")

print(f"\n• Commits por bug (con bugs): {productividad_bugs['avg_commits_per_bug']:.2f}")
if correlacion_commits_bugs > 0.3:
    print("• Más commits tienden a asociarse con más bugs (mayor complejidad)")
elif correlacion_commits_bugs < -0.3:
    print("• Más commits tienden a asociarse con menos bugs (mejor calidad)")
else:
    print("• No hay relación clara entre commits y bugs")

# Veredicto preliminar
print(f"\n=== VEREDICTO PRELIMINAR ===")
if abs(correlacion_bugs) < 0.1:
    print("HIPÓTESIS REFUTADA: Los bugs no tienen impacto significativo en el éxito")
elif correlacion_bugs < -0.1:
    print("HIPÓTESIS CONFIRMADA (débil): Más bugs se asocian ligeramente con menor éxito")
else:
    print("HIPÓTESIS INVERSA: Más bugs se asocian con mayor éxito (inesperado)")
```

## Veredicto esperado
- ✅ **CONFIRMADA (débil)** si correlación entre -0.1 y -0.3
- ❌ **REFUTADA** si correlación cercana a 0 (muy probable dado que mediana = 0)
- 🔄 **CONTEXTUAL** si la correlación es débil pero hay patrones interesantes en subgrupos

## ✅ Criterio de completado
1. Correlación calculada y mostrada con interpretación
2. Tres gráficos generados con títulos descriptivos y anotaciones
3. Estadísticas comparativas completas (promedio, mediana, min/max, desviación)
4. Análisis de patrones de bugs reportados
5. Veredicto claro con sustento numérico
6. **Cumplimiento del estándar de calidad**: Documentación detallada con explicaciones comprensibles

## 📋 Estándar de Calidad Obligatorio

**REQUISITO**: Este plan debe seguir el estándar de documentación detallada:

### Archivo de Estadísticas (`plan4_bugs_estadisticas.txt`):
- ✅ Metodología clara (variables, técnicas, herramientas)
- ✅ Definición de rangos de bugs con equivalencias prácticas
- ✅ Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
- ✅ Análisis por categorías con interpretación
- ✅ Insights clave: umbral de tolerancia, patrones de calidad
- ✅ Veredicto con contexto y métricas
- ✅ Recomendaciones prácticas basadas en evidencia
- ✅ Limitaciones del análisis documentadas

### Gráficos (3 requeridos):
- ✅ Títulos descriptivos con hipótesis
- ✅ Etiquetas claras en ejes
- ✅ Anotaciones estadísticas (medianas, porcentajes)
- ✅ Referencias visuales para rangos de bugs
- ✅ Equivalencias prácticas (ej: complejidad de código)

### Formato:
- Script: `plan4_bugs_analysis.py`
- Estadísticas: `plan4_bugs_estadisticas.txt`
- Gráficos: `plan4_bugs_boxplot.png`, `plan4_bugs_histograma.png`, `plan4_bugs_tasa_exito.png`
- Carpeta: `notebooks/results/plan4-bugs-reportados/`

**Sin excepción**: Plan no está "completado" hasta cumplir todos estos estándares.

## Archivos de salida sugeridos
- `notebooks/results/plan4-bugs-reportados/plan4_bugs_boxplot.png`
- `notebooks/results/plan4-bugs-reportados/plan4_bugs_histograma.png`
- `notebooks/results/plan4-bugs-reportados/plan4_bugs_tasa_exito.png`
- `notebooks/results/plan4-bugs-reportados/plan4_bugs_estadisticas.txt`
- `results/plan4_bugs_estadisticas.txt`
