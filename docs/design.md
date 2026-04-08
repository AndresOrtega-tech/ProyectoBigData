# Design — BigDataProject

> **Basado en:** init-pipeline (proyecto existente)
> **Tipo de proyecto:** Data Science / Análisis EDA — Académico
> **Última actualización:** generado por init_doc

---

## Arquitectura General

El proyecto sigue una arquitectura de **análisis de datos local por hipótesis**, sin servidor, sin base de datos externa y sin frontend framework. Cada plan de análisis es un módulo independiente que lee del mismo dataset CSV y escribe resultados en su propia carpeta.

```
[Dataset CSV]
     │
     ├──► [PySpark (procesamiento Big Data local)]
     │         │
     │         └──► [Resultados: .txt + .png por plan]
     │
     ├──► [Pandas / NumPy (análisis estadístico)]
     │         │
     │         └──► [Visualizaciones: Matplotlib / Seaborn]
     │
     └──► [Streamlit Dashboard (visualización interactiva)]
               │
               └──► [Deploy: Streamlit Cloud]
```

<!-- inferido del código -->

---

## Componentes Principales

### 1. Notebooks de Análisis (`notebooks/`)

| Archivo | Rol |
|---------|-----|
| `01_eda_inicial.ipynb` | Análisis exploratorio inicial del dataset completo |
| `plan1_cafeina_analysis.py` | Hipótesis cafeína vs task_success |
| `plan2_horas_codigo_analysis.py` | Hipótesis horas de código vs task_success |
| `plan3_cognitiva_analysis.py` | Hipótesis carga cognitiva vs task_success |
| `plan4_bugs_analysis.py` | Hipótesis bugs reportados vs task_success |
| `plan5_sueno_analysis.py` | Hipótesis sueño vs task_success |

Cada script de plan sigue el mismo patrón interno:
1. Inicializar sesión Spark local (`master("local[*]")`)
2. Cargar `data/ai_dev_productivity.csv`
3. Calcular correlación de Pearson entre la variable objetivo y `task_success`
4. Generar visualizaciones (boxplot, histograma, tasa de éxito, scatter/heatmap según el plan)
5. Escribir archivo de estadísticas con 8 secciones obligatorias
6. Guardar todo en `notebooks/results/planX-{nombre}/`

<!-- inferido del código -->

---

### 2. Dataset (`data/`)

Fuente única de verdad del proyecto. Archivo CSV local con 500 registros.

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `hours_coding` | float64 | Horas de programación por sesión |
| `coffee_intake_mg` | int64 | Miligramos de cafeína consumidos |
| `distractions` | int64 | Número de distracciones |
| `sleep_hours` | float64 | Horas de sueño la noche anterior |
| `commits` | int64 | Número de commits realizados |
| `bugs_reported` | int64 | Número de bugs reportados |
| `ai_usage_hours` | float64 | Horas de uso de herramientas IA |
| `cognitive_load` | float64 | Carga cognitiva auto-reportada (1–10) |
| `task_success` | int64 | Variable target: éxito de la tarea (0 o 1) |

> **Variable target**: `task_success` es la variable dependiente en todos los análisis.

---

### 3. Resultados por Plan (`notebooks/results/`)

Cada plan produce su propia carpeta con resultados estandarizados:

```
notebooks/results/
├── plan1-cafeina/
│   ├── plan1_cafeina_boxplot.png
│   ├── plan1_cafeina_histograma.png
│   ├── plan1_cafeina_tasa_exito.png
│   └── plan1_cafeina_estadisticas.txt
├── plan2-horas-codigo/
│   ├── plan2_horas_boxplot.png
│   ├── plan2_horas_histograma.png
│   ├── plan2_horas_tasa_exito.png
│   └── plan2_horas_estadisticas.txt
├── plan3-carga-cognitiva/
│   ├── plan3_cognitiva_boxplot.png
│   ├── plan3_cognitiva_tasa_exito.png
│   ├── plan3_cognitiva_scatter.png
│   ├── plan3_cognitiva_heatmap.png
│   └── plan3_cognitiva_estadisticas.txt
├── plan4-bugs-reportados/
│   ├── plan4_bugs_boxplot.png
│   ├── plan4_bugs_tasa_exito.png
│   ├── plan4_bugs_scatter.png
│   ├── plan4_bugs_categoria.png
│   └── plan4_bugs_estadisticas.txt
└── plan5-sueno/
    ├── plan5_sueno_boxplot.png
    ├── plan5_sueno_histograma.png
    ├── plan5_sueno_linea.png
    ├── plan5_sueno_heatmap.png
    ├── plan5_sueno_tasa_exito.png
    └── plan5_sueno_estadisticas.txt
```

**Convención de nombres:**
- Scripts: `planX_{hipotesis}_analysis.py`
- Estadísticas: `planX_{hipotesis}_estadisticas.txt`
- Gráficos: `planX_{hipotesis}_{tipo}.png`
- Carpetas: `notebooks/results/planX-{nombre}/`

---

### 4. Dashboard Streamlit (`dashboard/`)

Componente de visualización interactiva desplegado en **Streamlit Cloud**.

```
dashboard/
├── dashboard.py          # Aplicación principal
├── assets/               # 19 imágenes (copias de notebooks/results/)
│   └── plan1..5_*.png
├── data/
│   └── ai_dev_productivity.csv   # Copia local del dataset
└── README.md             # Instrucciones de deploy
```

**Secciones del dashboard:**

| Sección | Descripción |
|---------|-------------|
| Header | Título y descripción del proyecto |
| Resumen General | Tabla con las 5 hipótesis, correlaciones y veredictos |
| Plan 1–5 | Galerías individuales con visualizaciones e insights por plan |
| Análisis Comparativo | Barras de correlación, heatmap, ranking de impacto |
| Dataset Explorer | Tabla filtrable del CSV con estadísticas en tiempo real |

**Dependencias propias del dashboard:**

```
dashboard/requirements.txt  <!-- TODO: verificar si está separado del requirements.txt raíz -->
```

---

## Flujo de Datos

```
Usuario
   │
   ├── Jupyter Notebook (análisis exploratorio)
   │       │
   │       ▼
   │   data/ai_dev_productivity.csv
   │       │
   │       ├──► PySpark Session (local[*])
   │       │         └──► Correlación, agregaciones, estadísticas
   │       │
   │       └──► Pandas DataFrame
   │                 └──► Matplotlib / Seaborn → PNGs
   │
   ├── Script planX_analysis.py
   │       │
   │       ▼
   │   notebooks/results/planX-{nombre}/
   │       ├── estadisticas.txt
   │       └── *.png
   │
   └── Streamlit Dashboard
           │
           ▼
       dashboard/assets/  (imágenes copiadas manualmente)
       dashboard/data/    (CSV copiado manualmente)
           │
           ▼
       Streamlit Cloud (deploy público)
```

> **Punto de atención**: las imágenes en `dashboard/assets/` son copias manuales de `notebooks/results/`. No hay sincronización automática entre ambas carpetas. <!-- inferido del código -->

---

## Patrones Utilizados

| Patrón | Descripción |
|--------|-------------|
| **Notebook Pattern** | Análisis iterativo en celdas con visualización inmediata |
| **Hypothesis-per-module** | Cada hipótesis vive en su propio script independiente |
| **Hybrid Processing** | PySpark para Big Data + Pandas para análisis local |
| **Results-per-plan** | Resultados organizados en carpetas por plan, no mezclados |
| **Standardized Quality** | Todas las salidas siguen el mismo estándar de 8 secciones + 4 requisitos de gráficos |

---

## Decisiones de Arquitectura (ADRs)

### ADR-001 — Stack de análisis de datos
- **Decisión**: Python + Pandas + Jupyter como base
- **Alternativas descartadas**: R/RStudio, scripts .py puros, Excel
- **Consecuencia**: ecosistema rico, reproducible, con curva de aprendizaje inicial

### ADR-002 — PySpark local sin Hadoop
- **Decisión**: `SparkSession.builder.master("local[*]")`
- **Alternativas descartadas**: Hadoop completo, solo Pandas, Dask
- **Consecuencia**: simple de configurar, escalable a futuro, requiere Java 11

### ADR-003 — Planes de análisis independientes
- **Decisión**: un script por hipótesis en lugar de un notebook monolítico
- **Alternativas descartadas**: notebook único, sin estructura
- **Consecuencia**: organización clara, reproducibilidad, mayor cantidad de archivos

### ADR-004 — Resultados organizados por plan
- **Decisión**: `notebooks/results/planX-{nombre}/` como destino de outputs
- **Consecuencia**: fácil navegación, historial completo, comparación entre planes

### ADR-005 — Estándar de calidad obligatorio
- **Decisión**: 8 secciones en estadísticas + 4 requisitos en gráficos, sin excepciones
- **Consecuencia**: calidad consistente, mayor esfuerzo por plan, mejora la comprensión

---

## Infraestructura

| Componente | Entorno | Detalle |
|------------|---------|---------|
| Análisis | Local | Python venv, Jupyter |
| Dataset | Local | CSV en `data/` |
| Resultados | Local | `notebooks/results/` |
| Dashboard | Streamlit Cloud | Deploy público desde `dashboard/` |
| Versionado | <!-- TODO: verificar --> | No se detectó `.git` inicializado activamente |
| DevContainer | `.devcontainer/devcontainer.json` | Configuración para entorno reproducible <!-- TODO: verificar si está en uso --> |

---

## Limitaciones de Diseño Conocidas

- **Sin API**: no hay endpoints, todo el análisis es local y manual.
- **Sin automatización de sincronización**: las imágenes del dashboard se copian a mano desde `notebooks/results/`.
- **Dataset estático**: el CSV no se actualiza dinámicamente; cualquier cambio requiere reejecutar los scripts.
- **Sin tests automatizados**: no se detectaron tests unitarios ni de integración. <!-- inferido del código -->
- **Dependencia de Java 11**: PySpark no funciona sin Java instalado localmente.