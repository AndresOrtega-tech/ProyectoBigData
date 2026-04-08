# Project Brief — BigDataProject

> **Pipeline:** init-doc (proyecto existente)
> **Fecha de inicialización:** 2025
> **Confirmado por:** Andres

---

## Objetivo

Analizar la productividad de desarrolladores que utilizan herramientas de Inteligencia Artificial, identificando patrones y correlaciones entre variables de comportamiento (cafeína, horas de código, carga cognitiva, bugs, sueño) y el éxito en tareas de desarrollo.

## Problema que resuelve

No existe evidencia cuantitativa clara sobre qué factores del entorno y comportamiento de un desarrollador influyen más en el éxito de sus tareas diarias. Este proyecto construye esa evidencia mediante análisis estadístico y visualización de datos.

## Contexto del proyecto

- **Tipo:** Académico <!-- confirmado por Andres -->
- **Estado:** Completado en análisis (Planes 0–5), Dashboard desplegado en Streamlit Cloud <!-- confirmado por Andres -->
- **Cambios pendientes:** En curso — el usuario indicó que hay modificaciones próximas al proyecto <!-- confirmado por Andres -->

## Stack actual

| Tecnología | Versión | Rol |
|------------|---------|-----|
| Python | 3.14 | Lenguaje principal |
| PySpark | 3.5.0 | Procesamiento Big Data local (sin Hadoop) |
| Pandas | 3.0.0 | Manipulación y análisis de datos |
| NumPy | 2.4.2 | Computación numérica |
| Matplotlib | 3.10.8 | Visualización estática |
| Seaborn | 0.13.2 | Visualización estadística |
| Jupyter | 1.1.1 | Entorno de notebooks interactivo |
| Streamlit | latest | Dashboard interactivo desplegado |
| Java | 11.0.30 | Runtime requerido por PySpark |

<!-- inferido del código -->

## Dataset

**Archivo:** `data/ai_dev_productivity.csv`
**Registros:** 500
**Variables:** 9

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `hours_coding` | float64 | Horas de programación por sesión |
| `coffee_intake_mg` | int64 | Miligramos de cafeína consumidos |
| `distractions` | int64 | Número de distracciones |
| `sleep_hours` | float64 | Horas de sueño la noche anterior |
| `commits` | int64 | Número de commits realizados |
| `bugs_reported` | int64 | Número de bugs reportados |
| `ai_usage_hours` | float64 | Horas de uso de herramientas de IA |
| `cognitive_load` | float64 | Carga cognitiva auto-reportada (1–10) |
| `task_success` | int64 | Variable objetivo: éxito de la tarea (0=fallo, 1=éxito) |

## MVP real detectado

El MVP consiste en el análisis completo de 5 hipótesis sobre factores de productividad más un dashboard interactivo que consolida todos los resultados:

1. **Análisis EDA inicial** — exploración general del dataset
2. **5 planes de hipótesis** — cada uno con scripts PySpark, visualizaciones y estadísticas
3. **Dashboard Streamlit** — visualización interactiva con 19 gráficos + Dataset Explorer
4. **Estándar de calidad documentado** — 8 secciones por archivo de estadísticas, 4 requisitos por gráfico

## Restricciones conocidas

- Sin base de datos externa — todo el procesamiento es sobre archivos locales
- Sin API ni autenticación — proyecto de análisis offline
- Requiere Java 11 instalado para PySpark — dependencia crítica del entorno
- Dataset fijo de 500 registros — no hay ingesta de datos en tiempo real
- PySpark en modo local — no hay clúster distribuido real
- Sin tests automatizados — validación manual de resultados <!-- inferido del código -->

## Hipótesis analizadas y resultados

| Plan | Variable | Correlación | Veredicto |
|------|----------|-------------|-----------|
| Plan 1 | Cafeína vs Task Success | +0.695 | ✅ Confirmada |
| Plan 2 | Horas de Código vs Task Success | +0.616 | ✅ Confirmada |
| Plan 3 | Carga Cognitiva vs Task Success | -0.200 | ✅ Confirmada (débil) |
| Plan 4 | Bugs Reportados vs Task Success | -0.178 | ✅ Confirmada (muy débil) |
| Plan 5 | Sueño vs Task Success | +0.187 | ✅ Confirmada (débil) |

## Siguiente paso

Pipeline inicializado retroactivamente — continuar con `change-request-workflow.md` para nuevas features o con `verifier-archiver-workflow.md` para validar el estado actual.

Los cambios próximos mencionados por el usuario deben entrar por el flujo de change requests.