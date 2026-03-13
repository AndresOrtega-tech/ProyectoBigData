# CHANGELOG

Todos los cambios notables de este proyecto se documentarán en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Sin publicar]

### Añadido
- PySpark 3.5.0 para procesamiento Big Data local
- Java 11 como runtime para PySpark
- Planes estructurados de análisis en `.windsurf/plans/` (6 planes)
  - Plan 0: Configuración PySpark local (sin Hadoop)
  - Plan 1: Hipótesis Cafeína vs Task Success
  - Plan 2: Hipótesis Horas de Código vs Task Success
  - Plan 3: Hipótesis Carga Cognitiva vs Task Success
  - Plan 4: Hipótesis Bugs Reportados vs Task Success
  - Plan 5: Hipótesis Sueño vs Task Success
- Enfoque híbrido PySpark + pandas para análisis y visualización
- Script `notebooks/plan1_cafeina_analysis.py` con análisis completo de hipótesis
- Organización de resultados por plan en `notebooks/results/{plan-nombre}/`
- Dataset `ai_dev_productivity.csv` con 500 registros de productividad de desarrolladores
- Notebook `01_eda_inicial.ipynb` con análisis exploratorio inicial completo
- Análisis estadístico descriptivo de todas las variables
- Verificación de valores nulos (dataset completo)
- Cálculo de rangos, promedios y medianas para cada variable
- Configuración de entorno virtual Python 3.14
- Dependencias para análisis de datos (Pandas, NumPy, Matplotlib, Seaborn)
- Documentación inicial del proyecto (SPECS, ARCHITECTURE, CHANGELOG)
- Resultados del Plan 1: correlación 0.695, veredicto CONFIRMADA
- Visualizaciones del Plan 1: boxplot, histograma, tasa de éxito
- Estadísticas detalladas del Plan 1 en `plan1_cafeina_estadisticas.txt`
- Memory `bigdata-analysis-standards` con estándar de calidad obligatorio
- Estándar de calidad aplicado a todos los planes (0-5) con requisitos detallados
- ADR-005: Estándar de Calidad de Análisis Obligatorio
- Script `notebooks/plan2_horas_codigo_analysis.py` con análisis completo del Plan 2
- Resultados del Plan 2: correlación 0.616, veredicto CONFIRMADA
- Visualizaciones del Plan 2: boxplot, histograma, tasa de éxito
- Estadísticas detalladas del Plan 2 en `plan2_horas_estadisticas.txt`
- Script `notebooks/plan3_cognitiva_analysis.py` con análisis completo del Plan 3
- Resultados del Plan 3: correlación -0.200, veredicto CONFIRMADA (débil)
- Visualizaciones del Plan 3: boxplot, tasa éxito, scatter, heatmap
- Estadísticas detalladas del Plan 3 en `plan3_cognitiva_estadisticas.txt`
- Script `notebooks/plan4_bugs_analysis.py` con análisis completo del Plan 4
- Resultados del Plan 4: correlación -0.178, veredicto CONFIRMADA (muy débil)
- Visualizaciones del Plan 4: boxplot, tasa éxito, scatter, categoría
- Estadísticas detalladas del Plan 4 en `plan4_bugs_estadisticas.txt`
- Script `notebooks/plan5_sueno_analysis.py` con análisis completo del Plan 5
- Resultados del Plan 5: correlación 0.187, veredicto CONFIRMADA (débil)
- Visualizaciones del Plan 5: boxplot, tasa éxito, histograma, línea, heatmap
- Estadísticas detalladas del Plan 5 en `plan5_sueno_estadisticas.txt`

### Modificado
- Plan 0: Simplificado de Hadoop + PySpark a PySpark local únicamente
- Planes 1-5: Actualizados para usar archivos locales en lugar de HDFS
- Arquitectura: Actualizada para reflejar procesamiento híbrido PySpark + pandas
- Estructura de proyecto: Agregada carpeta `notebooks/results/` con organización por plan
- Plan 1: Mejorado con gráficos enriquecidos y estadísticas detalladas (111 líneas)
- Plan 2: Implementado con análisis completo y rendimientos decrecientes (121 líneas)
- Plan 3: Implementado con análisis de interacción y factores influyentes (118 líneas)
- Plan 4: Implementado con análisis de distribución y umbral crítico (181 líneas)
- Plan 5: Implementado con análisis de déficit y punto óptimo (267 líneas)
- Todos los planes: Agregado estándar de calidad obligatorio con 8 secciones requeridas

### Corregido
- Compatibilidad PySpark 3.5.0 con Java 11 (downgrade desde 4.1.1)
- Manejo de rutas relativas/absolutas para dataset desde directorio `notebooks/`

### Eliminado
- Dependencia de Hadoop/HDFS del flujo de análisis

---

## [0.1.0] - 2026-03-10

### Añadido
- Estructura inicial del proyecto
- Configuración básica de análisis de datos
