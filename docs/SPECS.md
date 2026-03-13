# SPECS - Especificaciones Técnicas

## Stack Tecnológico

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.14 | Lenguaje principal |
| PySpark | 3.5.0 | Procesamiento Big Data local |
| Pandas | 3.0.0 | Manipulación y análisis de datos |
| NumPy | 2.4.2 | Computación numérica |
| Jupyter | 1.1.1 | Entorno de notebooks |
| Matplotlib | 3.10.8 | Visualización |
| Seaborn | 0.13.2 | Visualización estadística |
| Java | 11.0.30 | Runtime para PySpark |

## Endpoints de la API

No se detectaron endpoints de API en este proyecto. Es un proyecto de análisis de datos local.

## Modelos de Base de Datos

No se detectó base de datos externa. El proyecto utiliza un archivo CSV local:

### Dataset: ai_dev_productivity.csv

| Columna | Tipo | Descripción |
|---------|------|-------------|
| hours_coding | float64 | Horas de coding por sesión |
| coffee_intake_mg | int64 | Miligramos de cafeína consumidos |
| distractions | int64 | Número de distracciones |
| sleep_hours | float64 | Horas de sueño |
| commits | int64 | Número de commits realizados |
| bugs_reported | int64 | Número de bugs reportados |
| ai_usage_hours | float64 | Horas de uso de herramientas IA |
| cognitive_load | float64 | Carga cognitiva (escala 1-10) |
| task_success | int64 | Éxito de la tarea (0 o 1) |

## Variables de Entorno

No se detectaron variables de entorno requeridas para este proyecto.

## Integraciones de Terceros

No se detectaron integraciones con servicios externos.

## Convenciones del Proyecto

- **Nomenclatura**: Archivos en inglés, comentarios en español
- **Estructura**: Datos en `/data/`, notebooks en `/notebooks/`, planes en `.windsurf/plans/`
- **Formato**: Jupyter notebooks para análisis exploratorio
- **Versionado**: Sin configuración Git detectada
- **Análisis**: Enfoque híbrido PySpark + pandas para procesamiento Big Data local
- **Resultados**: Organizados por plan en `notebooks/results/{plan-nombre}/`

## Estándar de Calidad de Análisis

**Requisito obligatorio para todos los planes**:

### Archivos de Estadísticas (8 secciones requeridas):
- ✅ Metodología clara (variables, técnicas, herramientas)
- ✅ Definición de rangos con equivalencias prácticas
- ✅ Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
- ✅ Análisis por categorías con interpretación
- ✅ Insights clave explicados con sustento cuantitativo
- ✅ Veredicto con contexto y métricas
- ✅ Recomendaciones prácticas basadas en evidencia
- ✅ Limitaciones del análisis documentadas

### Gráficos (4 requisitos):
- ✅ Títulos descriptivos con hipótesis
- ✅ Etiquetas claras en ejes
- ✅ Anotaciones estadísticas (medianas, porcentajes)
- ✅ Referencias visuales y equivalencias prácticas

### Formato Estándar:
- Scripts: `planX_{hipotesis}_analysis.py`
- Estadísticas: `planX_{hipotesis}_estadisticas.txt`
- Gráficos: `planX_{hipotesis}_{tipo}.png`
- Carpetas: `notebooks/results/planX-{nombre}/`

**Regla**: Sin excepción, un plan no está "completado" hasta cumplir todos estos estándares.

## Planes de Análisis Disponibles

- **Plan 0**: Configuración PySpark local (sin Hadoop) - ✅ Completado
- **Plan 1**: Hipótesis Cafeína vs Task Success - ✅ Completado (correlación: 0.695)
- **Plan 2**: Hipótesis Horas de Código vs Task Success - ✅ Completado (correlación: 0.616)
- **Plan 3**: Hipótesis Carga Cognitiva vs Task Success - ✅ Completado (correlación: -0.200)
- **Plan 4**: Hipótesis Bugs Reportados vs Task Success - ✅ Completado (correlación: -0.178)
- **Plan 5**: Hipótesis Sueño vs Task Success - ✅ Completado (correlación: 0.187)

## Resultados de Análisis

### Plan 1: Hipótesis Cafeína ✅
- **Correlación observada**: 0.695 (esperada: +0.70)
- **Veredicto**: ✅ CONFIRMADA
- **Archivos**: `notebooks/results/plan1-cafeina/`
  - `plan1_cafeina_boxplot.png` - Distribución con anotaciones
  - `plan1_cafeina_histograma.png` - Patrones con rangos visibles
  - `plan1_cafeina_tasa_exito.png` - Relación dose-respuesta
  - `plan1_cafeina_estadisticas.txt` - Análisis completo (111 líneas)
- **Insights clave**:
  - Umbral crítico: 0% éxito con <200mg cafeína
  - Rango óptimo: 83.6% éxito con >400mg
  - Consistencia: Menor variabilidad en grupo exitoso
  - Impacto: 8.5x más probabilidad de éxito en rango alto

### Plan 2: Hipótesis Horas de Código ✅
- **Correlación observada**: 0.616 (esperada: +0.62)
- **Veredicto**: ✅ CONFIRMADA
- **Archivos**: `notebooks/results/plan2-horas-codigo/`
  - `plan2_horas_boxplot.png` - Distribución con anotaciones de medianas
  - `plan2_horas_histograma.png` - Patrones con líneas de referencia de rangos
  - `plan2_horas_tasa_exito.png` - Tasa de éxito por rango con productividad
  - `plan2_horas_estadisticas.txt` - Análisis completo (121 líneas)
- **Insights clave**:
  - Umbral crítico: 0% éxito con <3 horas de código
  - Rango óptimo: 85.7% éxito con 6-9 horas
  - Rendimientos decrecientes: 76.9% éxito con >9 horas
  - Impacto: 69.4% más tiempo de código en grupo exitoso

### Plan 3: Hipótesis Carga Cognitiva ✅
- **Correlación observada**: -0.200 (esperada: -0.20)
- **Veredicto**: ✅ CONFIRMADA (débil)
- **Archivos**: `notebooks/results/plan3-carga-cognitiva/`
  - `plan3_cognitiva_boxplot.png` - Distribución con anotaciones de medianas
  - `plan3_cognitiva_tasa_exito.png` - Tasa de éxito por nivel de carga
  - `plan3_cognitiva_scatter.png` - Interacción carga vs horas con commits
  - `plan3_cognitiva_heatmap.png` - Tasa de éxito por combinación carga+horas
  - `plan3_cognitiva_estadisticas.txt` - Análisis completo (118 líneas)
- **Insights clave**:
  - Correlación negativa débil confirma hipótesis parcialmente
  - Factor más influyente: Sueño reduce carga cognitiva (r=-0.734)
  - Peor combinación: Carga alta + horas pocas = 0% éxito
  - Impacto relativo: Menor que cafeína (0.70) y horas (0.62)

### Plan 4: Hipótesis Bugs Reportados ✅
- **Correlación observada**: -0.178 (esperada: -0.18)
- **Veredicto**: ✅ CONFIRMADA (muy débil)
- **Archivos**: `notebooks/results/plan4-bugs-reportados/`
  - `plan4_bugs_boxplot.png` - Distribución por éxito con conteos
  - `plan4_bugs_tasa_exito.png` - Tasa de éxito por número exacto de bugs
  - `plan4_bugs_scatter.png` - Relación cantidad vs calidad con tendencia
  - `plan4_bugs_categoria.png` - Métricas comparativas por categoría
  - `plan4_bugs_estadisticas.txt` - Análisis completo (181 líneas)
- **Insights clave**:
  - Correlación muy débil (-0.178) casi exacta a esperada
  - 52.2% de sesiones sin bugs reportados (mediana = 0)
  - Umbral crítico: 4+ bugs = 0% éxito
  - Impacto relativo: El menor de todos los factores analizados

### Plan 5: Hipótesis Sueño ✅
- **Correlación observada**: 0.187 (débil positiva)
- **Veredicto**: ✅ CONFIRMADA (débil)
- **Archivos**: `notebooks/results/plan5-sueno/`
  - `plan5_sueno_boxplot.png` - Distribución con medianas y referencia 8h
  - `plan5_sueno_tasa_exito.png` - Tasa de éxito por nivel de descanso
  - `plan5_sueno_histograma.png` - Distribución con líneas de referencia
  - `plan5_sueno_linea.png` - Tasa de éxito por horas exactas con punto óptimo
  - `plan5_sueno_heatmap.png` - Interacción sueño + horas de código
  - `plan5_sueno_estadisticas.txt` - Análisis completo (267 líneas)
- **Insights clave**:
  - Correlación débil positiva (0.187) confirma hipótesis parcialmente
  - Punto óptimo: 7.1 horas de sueño con 92.3% éxito
  - Déficit severo (<5h) reduce éxito a 18.6%
  - Impacto relativo: Moderado comparado con otros factores

### Calidad de Análisis Implementada
- **Memory**: `bigdata-analysis-standards` con requisitos obligatorios
- **Aplicación**: Todos los planes (0-5) actualizados con estándar
- **Cumplimiento**: Plan 1, Plan 2, Plan 3, Plan 4 y Plan 5 cumplen 100% estándar como ejemplos a seguir
