# Plan 5: Hipótesis Sueño - Impacto del Descanso en el Éxito

Determinar si dormir menos de ciertas horas impacta negativamente el éxito e identificar el punto óptimo de descanso.

## Hipótesis
"Dormir menos de X horas impacta negativamente el task_success"
### Correlación: por confirmar

## Enfoque Híbrido
1. **PySpark** → Cálculo de correlaciones y análisis de rangos
2. **pandas** → Visualización de distribuciones y tendencias
3. **matplotlib/seaborn** → Gráficos para identificar puntos de corte

## Paso 1 — Configuración inicial

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("Hipotesis_Sueno") \
    .getOrCreate()

# Cargar dataset local
df_spark = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)
```

## Paso 2 — Análisis en PySpark

### 2.1 Promedio de sueño por task_success
```python
# Promedio de horas de sueño agrupado por éxito
promedio_sueno = df_spark.groupBy("task_success") \
    .agg(
        avg("sleep_hours").alias("avg_sleep"),
        avg("hours_coding").alias("avg_coding"),
        avg("cognitive_load").alias("avg_cognitive_load"),
        count("*").alias("total_registros")
    )

promedio_sueno.show()
```

### 2.2 Calcular correlación principal
```python
# Correlación sueño vs éxito
correlacion_sueno = df_spark.stat.corr("sleep_hours", "task_success")
print(f"Correlación sleep_hours vs task_success: {correlacion_sueno:.3f}")

# Correlación sueño vs otras variables
correlacion_sueno_horas = df_spark.stat.corr("sleep_hours", "hours_coding")
correlacion_sueno_cognitiva = df_spark.stat.corr("sleep_hours", "cognitive_load")
correlacion_sueno_cafeina = df_spark.stat.corr("sleep_hours", "coffee_intake_mg")

print(f"Correlación sleep_hours vs hours_coding: {correlacion_sueno_horas:.3f}")
print(f"Correlación sleep_hours vs cognitive_load: {correlacion_sueno_cognitiva:.3f}")
print(f"Correlación sleep_hours vs coffee_intake_mg: {correlacion_sueno_cafeina:.3f}")
```

### 2.3 Crear rangos de sueño
```python
# Crear rangos de sueño
df_con_rangos = df_spark.withColumn(
    "sleep_rango",
    when(col("sleep_hours") < 5, "insuficiente")
    .when((col("sleep_hours") >= 5) & (col("sleep_hours") < 6), "bajo")
    .when((col("sleep_hours") >= 6) & (col("sleep_hours") < 8), "optimo")
    .otherwise("alto")
)

# Calcular tasa de éxito por rango
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
```

### 2.4 Feature engineering: sleep_deficit
```python
# Calcular déficit de sueño (8 horas como referencia)
df_con_deficit = df_con_rangos.withColumn(
    "sleep_deficit",
    when(col("sleep_hours") < 8, 8 - col("sleep_hours")).otherwise(0)
)

# Correlación con déficit de sueño
correlacion_deficit = df_con_deficit.stat.corr("sleep_deficit", "task_success")
print(f"Correlación sleep_deficit vs task_success: {correlacion_deficit:.3f}")

# Análisis por nivel de déficit
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
```

### 2.5 Análisis por hora exacta
```python
# Tasa de éxito por hora exacta de sueño
horas_exito = df_spark.groupBy("sleep_hours") \
    .agg(
        avg("task_success").alias("tasa_exito"),
        count("*").alias("count")
    ) \
    .filter(col("count") >= 10) \  # Solo horas con suficientes datos
    .orderBy("sleep_hours")

horas_exito.show()
```

### 2.6 Análisis de interacción: sueño + horas de código
```python
# Crear combinación de rangos sueño + horas
df_con_interaccion = df_con_deficit.withColumn(
    "hours_rango",
    when(col("hours_coding") < 3, "pocas")
    .when((col("hours_coding") >= 3) & (col("hours_coding") < 6), "moderadas")
    .when((col("hours_coding") >= 6) & (col("hours_coding") < 9), "muchas")
    .otherwise("excesivas")
)

# Análisis de combinación sueño + horas
interaccion = df_con_interaccion.groupBy("sleep_rango", "hours_rango") \
    .agg(
        avg("task_success").alias("tasa_exito"),
        avg("cognitive_load").alias("avg_cognitive_load"),
        count("*").alias("total_registros")
    ) \
    .filter(col("total_registros") >= 5) \
    .orderBy("tasa_exito", ascending=False)

interaccion.show()
```

## Paso 3 — Convertir a pandas para visualización

```python
# Convertir resultados a pandas
promedio_sueno_pd = promedio_sueno.toPandas()
tasa_exito_rango_pd = tasa_exito_rango.toPandas()
deficit_stats_pd = deficit_stats.toPandas()
horas_exito_pd = horas_exito.toPandas()
interaccion_pd = interaccion.toPandas()
df_completo_pd = df_spark.toPandas()

# Detener Spark
spark.stop()
```

## Paso 4 — Visualizaciones

### Gráfico 1: Boxplot de horas de sueño por task_success
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_completo_pd, x="task_success", y="sleep_hours")
plt.title("Distribución de Horas de Sueño por Éxito de Tarea")
plt.xlabel("Éxito de Tarea (0=No, 1=Sí)")
plt.ylabel("Horas de Sueño")
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 2: Barplot de tasa de éxito por rango de sueño
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=tasa_exito_rango_pd, x="sleep_rango", y="tasa_exito")
plt.title("Tasa de Éxito por Rango de Sueño")
plt.xlabel("Rango de Sueño")
plt.ylabel("Tasa de Éxito")
plt.ylim(0, 1)

# Agregar etiquetas de valores
for i, row in tasa_exito_rango_pd.iterrows():
    plt.text(i, row.tasa_exito + 0.01, f"{row.tasa_exito:.2f}", 
             ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 3: Histograma de horas de sueño coloreado por task_success
```python
plt.figure(figsize=(12, 6))
sns.histplot(data=df_completo_pd, x="sleep_hours", hue="task_success", 
             bins=15, alpha=0.7, kde=True)
plt.title("Distribución de Horas de Sueño por Éxito")
plt.xlabel("Horas de Sueño")
plt.ylabel("Frecuencia")
plt.legend(title="Éxito", labels=["No", "Sí"])
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 4: Línea de tendencia de éxito por horas exactas de sueño
```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=horas_exito_pd, x="sleep_hours", y="tasa_exito", 
             marker='o', markersize=8)
plt.title("Tasa de Éxito por Horas Exactas de Sueño")
plt.xlabel("Horas de Sueño")
plt.ylabel("Tasa de Éxito")
plt.grid(True, alpha=0.3)

# Agregar línea de referencia para 8 horas
plt.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='8 horas (referencia)')
plt.legend()
plt.show()
```

### Gráfico 5: Heatmap de interacción sueño + horas de código
```python
# Pivot para heatmap
heatmap_data = interaccion_pd.pivot(index="sleep_rango", 
                                  columns="hours_rango", 
                                  values="tasa_exito")

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r", 
           vmin=0, vmax=1, cbar_kws={'label': 'Tasa de Éxito'})
plt.title("Tasa de Éxito: Sueño vs Horas de Código")
plt.xlabel("Rango de Horas de Código")
plt.ylabel("Rango de Sueño")
plt.tight_layout()
plt.show()
```

## Paso 5 — Análisis de resultados

### Información de valor a extraer:
- **Correlación real**: Fuerza y dirección de la relación sueño-éxito
- **Punto de corte**: ¿Cuántas horas son "suficientes"?
- **Rango óptimo**: ¿Dónde está el sweet spot de descanso?
- **Exceso de sueño**: ¿Dormir demasiado también es negativo?

### Código para análisis final:
```python
print("=== ESTADÍSTICAS DE SUEÑO ===")
print(f"Correlación con task_success: {correlacion_sueno:.3f}")
print(f"Correlación con sleep_deficit: {correlacion_deficit:.3f}")

for success in [0, 1]:
    subset = df_completo_pd[df_completo_pd['task_success'] == success]
    print(f"\nTask Success = {success}:")
    print(f"  Promedio sueño: {subset['sleep_hours'].mean():.1f} h")
    print(f"  Mediana sueño: {subset['sleep_hours'].median():.1f} h")

print("\n=== ANÁLISIS POR RANGO DE SUEÑO ===")
for _, row in tasa_exito_rango_pd.iterrows():
    print(f"Rango '{row['sleep_rango']}':")
    print(f"  Tasa éxito: {row['tasa_exito']:.1%}")
    print(f"  Horas promedio: {row['avg_hours']:.1f} h")
    print(f"  Carga cognitiva: {row['avg_cognitive_load']:.1f}")

print("\n=== ANÁLISIS POR DÉFICIT DE SUEÑO ===")
for _, row in deficit_stats_pd.iterrows():
    print(f"Rango '{row['deficit_rango']}':")
    print(f"  Tasa éxito: {row['tasa_exito']:.1%}")
    print(f"  Sueño promedio: {row['avg_sleep']:.1f} h")

# Identificar punto de corte
rango_optimo = tasa_exito_rango_pd.loc[tasa_exito_rango_pd['tasa_exito'].idxmax()]
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
```

## Veredicto esperado
- ✅ **CONFIRMADA** si correlación > 0.3 y hay un punto de corte claro
- ❌ **REFUTADA** si correlación < 0.1 o no hay patrón claro
- 🔄 **PARCIAL** si hay correlación moderada pero con matices (ej: demasiado sueño también es malo)

## ✅ Criterio de completado
1. Correlación calculada y mostrada con interpretación
2. Tres gráficos generados con títulos descriptivos y anotaciones
3. Estadísticas comparativas completas (promedio, mediana, min/max, desviación)
4. Análisis de patrones de sueño y rendimiento
5. Veredicto claro con sustento numérico
6. **Cumplimiento del estándar de calidad**: Documentación detallada con explicaciones comprensibles

## 📋 Estándar de Calidad Obligatorio

**REQUISITO**: Este plan debe seguir el estándar de documentación detallada:

### Archivo de Estadísticas (`plan5_sueno_estadisticas.txt`):
- ✅ Metodología clara (variables, técnicas, herramientas)
- ✅ Definición de rangos de sueño con equivalencias prácticas
- ✅ Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
- ✅ Análisis por categorías con interpretación
- ✅ Insights clave: umbral óptimo, patrones de descanso
- ✅ Veredicto con contexto y métricas
- ✅ Recomendaciones prácticas basadas en evidencia
- ✅ Limitaciones del análisis documentadas

### Gráficos (3 requeridos):
- ✅ Títulos descriptivos con hipótesis
- ✅ Etiquetas claras en ejes
- ✅ Anotaciones estadísticas (medianas, porcentajes)
- ✅ Referencias visuales para rangos de sueño
- ✅ Equivalencias prácticas (ej: ciclos de sueño)

### Formato:
- Script: `plan5_sueno_analysis.py`
- Estadísticas: `plan5_sueno_estadisticas.txt`
- Gráficos: `plan5_sueno_boxplot.png`, `plan5_sueno_histograma.png`, `plan5_sueno_tasa_exito.png`
- Carpeta: `notebooks/results/plan5-sueno/`

**Sin excepción**: Plan no está "completado" hasta cumplir todos estos estándares.

## Archivos de salida sugeridos
- `notebooks/results/plan5-sueno/plan5_sueno_boxplot.png`
- `notebooks/results/plan5-sueno/plan5_sueno_histograma.png`
- `notebooks/results/plan5-sueno/plan5_sueno_tasa_exito.png`
- `notebooks/results/plan5-sueno/plan5_sueno_estadisticas.txt`
- `results/plan5_sueno_interaccion.png`
- `results/plan5_sueno_estadisticas.txt`
