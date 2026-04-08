# Plan 1: Hipótesis Cafeína - Análisis de Correlación con Task Success

Analizar la relación entre el consumo de cafeína y el éxito en las tareas de desarrollo utilizando PySpark para procesamiento y pandas/matplotlib para visualización.

## Hipótesis
"Mayor consumo de cafeína está asociado con mayor task_success"
### Correlación esperada: +0.70

## Enfoque Híbrido
1. **PySpark** → Cargar datos, filtrar, agrupar, calcular estadísticas
2. **pandas** → Convertir resultado de PySpark a pandas para gráficos (`.toPandas()`)
3. **matplotlib/seaborn** → Generar las visualizaciones

## Paso 1 — Configuración inicial

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("Hipotesis_Cafeina") \
    .getOrCreate()

# Cargar dataset local
df_spark = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)
```

## Paso 2 — Análisis en PySpark

### 2.1 Calcular promedio de cafeína por task_success
```python
# Promedio de cafeína agrupado por éxito
promedio_cafeina = df_spark.groupBy("task_success") \
    .agg(avg("coffee_intake_mg").alias("avg_coffee_intake"),
         count("*").alias("total_registros"))

promedio_cafeina.show()
```

### 2.2 Calcular correlación
```python
# Correlación entre cafeína y éxito
correlacion = df_spark.stat.corr("coffee_intake_mg", "task_success")
print(f"Correlación cafeína vs task_success: {correlacion:.3f}")
```

### 2.3 Crear rangos de cafeína
```python
from pyspark.sql.functions import col, when

# Crear rangos de cafeína
df_con_rangos = df_spark.withColumn(
    "coffee_rango",
    when(col("coffee_intake_mg") < 200, "bajo")
    .when((col("coffee_intake_mg") >= 200) & (col("coffee_intake_mg") <= 400), "medio")
    .otherwise("alto")
)

# Calcular tasa de éxito por rango
tasa_exito_rango = df_con_rangos.groupBy("coffee_rango") \
    .agg(
        avg("task_success").alias("tasa_exito"),
        count("*").alias("total_registros")
    ) \
    .orderBy("tasa_exito", ascending=False)

tasa_exito_rango.show()
```

## Paso 3 — Convertir a pandas para visualización

```python
# Convertir resultados a pandas
promedio_cafeina_pd = promedio_cafeina.toPandas()
tasa_exito_rango_pd = tasa_exito_rango.toPandas()
df_completo_pd = df_spark.toPandas()

# Detener Spark
spark.stop()
```

## Paso 4 — Visualizaciones

### Gráfico 1: Boxplot de cafeína por task_success
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_completo_pd, x="task_success", y="coffee_intake_mg")
plt.title("Distribución de Cafeína por Éxito de Tarea")
plt.xlabel("Éxito de Tarea (0=No, 1=Sí)")
plt.ylabel("Consumo de Cafeína (mg)")
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 2: Histograma de cafeína coloreado por task_success
```python
plt.figure(figsize=(12, 6))
sns.histplot(data=df_completo_pd, x="coffee_intake_mg", hue="task_success", 
             bins=20, alpha=0.7, kde=True)
plt.title("Distribución de Consumo de Cafeína por Éxito")
plt.xlabel("Consumo de Cafeína (mg)")
plt.ylabel("Frecuencia")
plt.legend(title="Éxito", labels=["No", "Sí"])
plt.grid(True, alpha=0.3)
plt.show()
```

### Gráfico 3: Barplot de tasa de éxito por rango de cafeína
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=tasa_exito_rango_pd, x="coffee_rango", y="tasa_exito")
plt.title("Tasa de Éxito por Rango de Cafeína")
plt.xlabel("Rango de Cafeína")
plt.ylabel("Tasa de Éxito")
plt.ylim(0, 1)

# Agregar etiquetas de valores
for i, row in tasa_exito_rango_pd.iterrows():
    plt.text(i, row.tasa_exito + 0.01, f"{row.tasa_exito:.2f}", 
             ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.show()
```

## Paso 5 — Análisis de resultados

### Información de valor a extraer:
- **Correlación numérica**: Valor exacto de la correlación
- **Promedio de cafeína**: Diferencia entre éxitos vs fracasos
- **Tasa de éxito por rango**: Porcentaje de éxito en cada nivel de consumo
- **Insights**: ¿Existe un punto óptimo de consumo?

### Código para análisis final:
```python
# Estadísticas descriptivas
print("=== ESTADÍSTICAS DE CAFEÍNA ===")
print(f"Correlación con task_success: {correlacion:.3f}")

for success in [0, 1]:
    subset = df_completo_pd[df_completo_pd['task_success'] == success]
    print(f"\nTask Success = {success}:")
    print(f"  Promedio cafeína: {subset['coffee_intake_mg'].mean():.1f} mg")
    print(f"  Mediana cafeína: {subset['coffee_intake_mg'].median():.1f} mg")
    print(f"  Desviación estándar: {subset['coffee_intake_mg'].std():.1f} mg")

print("\n=== TASA DE ÉXITO POR RANGO ===")
for _, row in tasa_exito_rango_pd.iterrows():
    print(f"Rango {row['coffee_rango']}: {row['tasa_exito']:.1%} de éxito ({row['total_registros']} registros)")
```

## Veredicto esperado
- ✅ **CONFIRMADA** si correlación > 0.5 y diferencias claras entre grupos
- ❌ **REFUTADA** si correlación < 0.3 o sin diferencias significativas

## ✅ Criterio de completado
1. Correlación calculada y mostrada
2. Tres gráficos generados y guardados
3. Estadísticas comparativas entre grupos
4. Conclusión clara con sustento numérico

## Archivos de salida sugeridos
- `results/plan1_cafeina_correlacion.png`
- `results/plan1_cafeina_histograma.png`
- `results/plan1_cafeina_tasa_exito.png`
- `results/plan1_cafeina_estadisticas.txt`
