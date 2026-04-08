# Spec — BigDataProject

> **Basado en:** init-pipeline (proyecto existente)
> **Tipo:** Data Science / EDA Académico
> **Estado:** En desarrollo activo — 6 planes completados, cambios próximos

---

## 1. Objetivo funcional

El sistema permite analizar correlaciones entre variables de comportamiento de desarrolladores (cafeína, horas de código, sueño, carga cognitiva, bugs) y su tasa de éxito en tareas (`task_success`), usando PySpark para procesamiento y Streamlit para visualización interactiva de resultados.

---

## 2. Usuario final

Estudiante universitario / investigador académico que ejecuta análisis de hipótesis de forma local y comparte resultados vía dashboard público en Streamlit Cloud.

---

## 3. Dataset

### Fuente
- Archivo: `data/ai_dev_productivity.csv`
- Registros: 500
- Sin base de datos externa — procesamiento completamente local <!-- inferido del código -->

### Esquema de variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `hours_coding` | float64 | Horas de programación por sesión |
| `coffee_intake_mg` | int64 | Miligramos de cafeína consumidos |
| `distractions` | int64 | Número de distracciones durante el trabajo |
| `sleep_hours` | float64 | Horas de sueño la noche anterior |
| `commits` | int64 | Número de commits realizados |
| `bugs_reported` | int64 | Número de bugs reportados |
| `ai_usage_hours` | float64 | Horas de uso de herramientas de IA |
| `cognitive_load` | float64 | Carga cognitiva auto-reportada (escala 1–10) |
| `task_success` | int64 | Variable objetivo: éxito de la tarea (0 = fallo, 1 = éxito) |

---

## 4. Features implementadas

### 4.1 EDA Inicial
- **Archivo:** `notebooks/01_eda_inicial.ipynb`
- **Descripción:** Análisis exploratorio inicial sobre el dataset completo
- **Incluye:** Estadísticas descriptivas, distribuciones, correlaciones generales
- **Estado:** ✅ Implementado

### 4.2 Plan 0 — Configuración PySpark local
- **Estado:** ✅ Implementado
- **Descripción:** Configura sesión Spark en modo local sin dependencia de Hadoop (`master("local[*]")`)
- **Resultado:** Entorno funcional para todos los planes posteriores

### 4.3 Plan 1 — Hipótesis Cafeína vs Task Success
- **Archivo:** `notebooks/plan1_cafeina_analysis.py`
- **Variable analizada:** `coffee_intake_mg`
- **Correlación observada:** `0.695` (esperada: `+0.70`)
- **Veredicto:** ✅ CONFIRMADA
- **Insights clave:**
  - Umbral crítico: 0% éxito con `<200mg`
  - Rango óptimo: 83.6% éxito con `>400mg`
  - 8.5x más probabilidad de éxito en rango alto
- **Resultados:** `notebooks/results/plan1-cafeina/` (3 gráficos + estadísticas)

### 4.4 Plan 2 — Hipótesis Horas de Código vs Task Success
- **Archivo:** `notebooks/plan2_horas_codigo_analysis.py`
- **Variable analizada:** `hours_coding`
- **Correlación observada:** `0.616` (esperada: `+0.62`)
- **Veredicto:** ✅ CONFIRMADA
- **Insights clave:**
  - Umbral crítico: 0% éxito con `<3h`
  - Rango óptimo: 85.7% éxito con `6–9h`
  - Rendimientos decrecientes: 76.9% éxito con `>9h`
- **Resultados:** `notebooks/results/plan2-horas-codigo/` (3 gráficos + estadísticas)

### 4.5 Plan 3 — Hipótesis Carga Cognitiva vs Task Success
- **Archivo:** `notebooks/plan3_cognitiva_analysis.py`
- **Variable analizada:** `cognitive_load`
- **Correlación observada:** `-0.200` (esperada: `-0.20`)
- **Veredicto:** ✅ CONFIRMADA (débil)
- **Insights clave:**
  - Correlación negativa débil confirmada
  - Factor más influyente sobre carga cognitiva: sueño (`r = -0.734`)
  - Peor combinación: carga alta + pocas horas = 0% éxito
- **Resultados:** `notebooks/results/plan3-carga-cognitiva/` (4 gráficos + estadísticas)

### 4.6 Plan 4 — Hipótesis Bugs Reportados vs Task Success
- **Archivo:** `notebooks/plan4_bugs_analysis.py`
- **Variable analizada:** `bugs_reported`
- **Correlación observada:** `-0.178` (esperada: `-0.18`)
- **Veredicto:** ✅ CONFIRMADA (muy débil)
- **Insights clave:**
  - 52.2% de sesiones con cero bugs reportados
  - Umbral crítico: `4+` bugs = 0% éxito
  - Correlación más débil de todos los factores analizados
- **Resultados:** `notebooks/results/plan4-bugs-reportados/` (4 gráficos + estadísticas)

### 4.7 Plan 5 — Hipótesis Sueño vs Task Success
- **Archivo:** `notebooks/plan5_sueno_analysis.py`
- **Variable analizada:** `sleep_hours`
- **Correlación observada:** `0.187` (débil positiva)
- **Veredicto:** ✅ CONFIRMADA (débil)
- **Insights clave:**
  - Punto óptimo: `7.1h` de sueño con 92.3% éxito
  - Déficit severo (`<5h`): reduce éxito a 18.6%
  - Interacción importante con horas de código (heatmap)
- **Resultados:** `notebooks/results/plan5-sueno/` (5 gráficos + estadísticas)

### 4.8 Dashboard Streamlit
- **Archivo principal:** `dashboard/dashboard.py`
- **Deploy:** Streamlit Cloud <!-- confirmado por Andres -->
- **Assets:** `dashboard/assets/` — 19 visualizaciones (copias de notebooks/results/)
- **Dataset embed:** `dashboard/data/ai_dev_productivity.csv`
- **Funcionalidades:**
  - Navegación por sidebar entre secciones
  - Galería visual por plan (todas las imágenes)
  - Tabla comparativa de hipótesis con correlaciones y veredictos
  - Dataset Explorer con filtros interactivos
  - Análisis comparativo entre variables
- **Estado:** ✅ Implementado y desplegado

---

## 5. Reglas de negocio — Estándar de calidad obligatorio

Todo plan de análisis DEBE cumplir los siguientes requisitos antes de considerarse completado:

### 5.1 Archivos de estadísticas (8 secciones requeridas)
1. Metodología clara (variables, técnicas, herramientas usadas)
2. Definición de rangos con equivalencias prácticas
3. Estadísticas descriptivas completas (promedio, mediana, min/max, desviación)
4. Análisis por categorías con interpretación
5. Insights clave explicados con sustento cuantitativo
6. Veredicto con contexto y métricas
7. Recomendaciones prácticas basadas en evidencia
8. Limitaciones del análisis documentadas

### 5.2 Gráficos (4 requisitos mínimos)
1. Títulos descriptivos que incluyan la hipótesis
2. Etiquetas claras en ambos ejes
3. Anotaciones estadísticas (medianas, porcentajes, líneas de referencia)
4. Referencias visuales y equivalencias prácticas

### 5.3 Nomenclatura estándar de archivos
| Tipo | Formato |
|------|---------|
| Script | `planX_{hipotesis}_analysis.py` |
| Estadísticas | `planX_{hipotesis}_estadisticas.txt` |
| Gráficos | `planX_{hipotesis}_{tipo}.png` |
| Carpeta de resultados | `notebooks/results/planX-{nombre}/` |

> **Regla sin excepción:** un plan no está "completado" hasta cumplir los tres estándares anteriores.

---

## 6. Resumen de correlaciones detectadas

| Plan | Variable | Correlación | Veredicto |
|------|----------|-------------|-----------|
| 1 | `coffee_intake_mg` | `+0.695` | ✅ Confirmada (fuerte) |
| 2 | `hours_coding` | `+0.616` | ✅ Confirmada (moderada) |
| 3 | `cognitive_load` | `-0.200` | ✅ Confirmada (débil) |
| 5 | `sleep_hours` | `+0.187` | ✅ Confirmada (débil) |
| 4 | `bugs_reported` | `-0.178` | ✅ Confirmada (muy débil) |

> Variable `ai_usage_hours` presente en el dataset pero no analizada en los planes actuales. <!-- TODO: verificar si está planificado un análisis específico -->

---

## 7. Cambios en curso

> El usuario indicó que hay cambios planificados sobre el proyecto. Los detalles se documentarán en `tasks.md` a medida que se definan. <!-- confirmado por Andres -->