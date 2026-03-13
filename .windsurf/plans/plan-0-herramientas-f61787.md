# Plan 0: Preparación de Herramientas - PySpark Local

Configurar el ambiente local con PySpark para procesamiento Big Data del dataset de productividad de desarrolladores con IA.

## Objetivo
Tener el ambiente listo para trabajar: PySpark + pandas. Esto cubre los puntos de análisis de datos con Spark sin necesidad de Hadoop.

## Paso 1 — Verificar Java

### Prerrequisitos
- Java 8 o 11 instalado (ya instalado con openjdk@11)

### Verificar instalación
```bash
java -version
# Debe mostrar openjdk version "11.0.30"
```

## Paso 2 — Instalar PySpark

```bash
# Instalar PySpark
pip install pyspark

# Verificar instalación
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test').getOrCreate()
print(f'PySpark version: {spark.version}')
spark.stop()
"
```

## Paso 3 — Cargar dataset en PySpark

```python
from pyspark.sql import SparkSession

# Crear sesión Spark (modo local, sin Hadoop)
spark = SparkSession.builder \
    .appName("AI_Dev_Productivity") \
    .master("local[*]") \
    .config("spark.sql.warehouse.dir", "spark-warehouse") \
    .getOrCreate()

# Cargar dataset local
df_spark = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)

# Mostrar información básica
print("Dataset cargado en PySpark:")
df_spark.show(5)
print("\nSchema:")
df_spark.printSchema()
print(f"\nTotal de registros: {df_spark.count()}")

# Detener Spark
spark.stop()
```

## Paso 4 — Verificación final

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Verificacion_Final") \
    .master("local[*]") \
    .getOrCreate()

# Leer dataset local
df_local = spark.read.csv("data/ai_dev_productivity.csv", 
                         header=True, inferSchema=True)

print("Dataset leído localmente:")
df_local.show(5)
print(f"Registros: {df_local.count()}")

# Verificar operaciones básicas
print("\nEstadísticas básicas:")
df_local.describe().show()

spark.stop()
```

## Paso 5 — Validar operaciones PySpark

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

spark = SparkSession.builder \
    .appName("Validacion_Operaciones") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv("data/ai_dev_productivity.csv", header=True, inferSchema=True)

# Probar agrupación y agregación
print("=== Agrupación por task_success ===")
df.groupBy("task_success").agg(avg("coffee_intake_mg")).show()

# Probar correlación
print("\n=== Correlación ===")
print(f"Correlación coffee_intake_mg vs task_success: {df.stat.corr('coffee_intake_mg', 'task_success'):.3f}")

spark.stop()
```

## Evidencias requeridas
- Captura de PySpark corriendo y mostrando versión
- Captura del dataset cargado en PySpark con schema visible
- Captura de operaciones básicas (groupby, correlación) funcionando

## ✅ Criterio de completado
Puedes ejecutar `df_spark.show()` y ver los datos sin errores, y las operaciones de agregación funcionan correctamente.

## 📋 Estándar de Calidad Requerido

**IMPORTANTE**: Todos los análisis siguientes deben seguir el estándar de documentación detallada:

### Para Archivos de Estadísticas:
- ✅ Metodología clara (variables, técnicas, herramientas)
- ✅ Definición de rangos con equivalencias prácticas
- ✅ Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
- ✅ Análisis por categorías con interpretación
- ✅ Insights clave explicados con sustento cuantitativo
- ✅ Veredicto con contexto y métricas
- ✅ Recomendaciones prácticas basadas en evidencia
- ✅ Limitaciones del análisis documentadas

### Para Gráficos:
- ✅ Títulos descriptivos con hipótesis
- ✅ Etiquetas claras en ejes
- ✅ Anotaciones estadísticas (medianas, porcentajes)
- ✅ Referencias visuales (líneas de rangos)
- ✅ Equivalencias prácticas (ej: tazas de café)

### Formato de Archivos:
- Scripts: `planX_{hipotesis}_analysis.py`
- Estadísticas: `planX_{hipotesis}_estadisticas.txt`
- Gráficos: `planX_{hipotesis}_{tipo}.png`
- Carpeta: `notebooks/results/planX-{nombre}/`

**Sin excepción**: Todo plan debe cumplir estos estándares para ser considerado "completado".

## Ventajas de este enfoque
- Sin complejidad de Hadoop
- Más rápido para datasets pequeños/medianos
- Ideal para desarrollo y prototipado
- Mismas capacidades de análisis que con Hadoop
