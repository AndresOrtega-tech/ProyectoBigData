# Blueprint: BigDataProject

> **Basado en:** init-pipeline (proyecto existente)
> **Tipo:** Data Science / Análisis Exploratorio Académico
> **Estado:** Análisis de hipótesis completado — dashboard desplegado en Streamlit Cloud

---

## Visión Integrada

BigDataProject es un proyecto académico de análisis exploratorio de datos (EDA) que estudia los factores que influyen en la productividad de desarrolladores que utilizan herramientas de inteligencia artificial. El sistema combina procesamiento Big Data local con PySpark y visualización interactiva mediante Streamlit, siguiendo un pipeline de análisis estructurado por hipótesis.

El proyecto completó exitosamente 5 planes de análisis sobre un dataset de 500 registros con 9 variables, validando todas las hipótesis planteadas (con distintos niveles de correlación) y expone los resultados a través de un dashboard público en Streamlit Cloud.

### Alcance real del MVP

| Componente | Estado |
|-----------|--------|
| EDA inicial del dataset | ✅ Implementado |
| Plan 0 — Configuración PySpark local | ✅ Implementado |
| Plan 1 — Hipótesis Cafeína | ✅ Implementado |
| Plan 2 — Hipótesis Horas de Código | ✅ Implementado |
| Plan 3 — Hipótesis Carga Cognitiva | ✅ Implementado |
| Plan 4 — Hipótesis Bugs Reportados | ✅ Implementado |
| Plan 5 — Hipótesis Sueño | ✅ Implementado |
| Dashboard Streamlit con 19 visualizaciones | ✅ Implementado |
| Dataset Explorer con filtros | ✅ Implementado |
| Deploy en Streamlit Cloud | ✅ Activo |

---

## Arquitectura Integrada

```
[Dataset CSV]
     │
     ▼
[PySpark (local[*])]  ←→  [Pandas / NumPy]
     │                          │
     ▼                          ▼
[Scripts análisis .py]   [EDA Jupyter Notebook]
     │
     ▼
[notebooks/results/{plan}/]
  ├── estadisticas.txt
  └── *.png (visualizaciones)
     │
     ▼
[Dashboard Streamlit]
  ├── assets/ (19 imágenes)
  ├── data/ (dataset)
  └── dashboard.py
     │
     ▼
[Streamlit Cloud — Deploy público]
```

### Decisiones de diseño clave

| Decisión | Motivo | ADR |
|---------|--------|-----|
| PySpark en modo local (sin Hadoop) | Evitar complejidad de cluster; dataset de 500 registros no lo requiere | ADR-002 |
| Procesamiento híbrido PySpark + Pandas | PySpark para escalabilidad, Pandas para visualización local | ADR-002 |
| Planes separados por hipótesis | Reproducibilidad y organización clara por objetivo analítico | ADR-003 |
| Resultados en carpetas por plan | Separación de artefactos, historial completo, comparación entre planes | ADR-004 |
| Estándar de calidad obligatorio | Consistencia en explicaciones y calidad de análisis académico | ADR-005 |

---

## Spec Resumida

### Dataset

**Fuente:** `data/ai_dev_productivity.csv` — 500 registros, 9 variables numéricas  
**Variable objetivo:** `task_success` (binaria: 0 = fallo, 1 = éxito)

| Variable | Tipo | Rol |
|---------|------|-----|
| `hours_coding` | float | Predictor principal (Plan 2) |
| `coffee_intake_mg` | int | Predictor principal (Plan 1) |
| `sleep_hours` | float | Predictor principal (Plan 5) |
| `cognitive_load` | float | Predictor principal (Plan 3) |
| `bugs_reported` | int | Predictor principal (Plan 4) |
| `distractions` | int | Variable auxiliar |
| `commits` | int | Variable auxiliar |
| `ai_usage_hours` | float | Variable auxiliar |
| `task_success` | int | Variable objetivo |

### Resultados consolidados

| Plan | Variable | Correlación | Veredicto |
|------|----------|-------------|-----------|
| 1 | Cafeína | +0.695 | ✅ CONFIRMADA |
| 2 | Horas de código | +0.616 | ✅ CONFIRMADA |
| 3 | Carga cognitiva | −0.200 | ✅ CONFIRMADA (débil) |
| 4 | Bugs reportados | −0.178 | ✅ CONFIRMADA (muy débil) |
| 5 | Horas de sueño | +0.187 | ✅ CONFIRMADA (débil) |

### Reglas de negocio y estándar de calidad

Todo plan de análisis debe cumplir **sin excepción**:

- **8 secciones** en archivos de estadísticas: metodología, rangos, estadísticas descriptivas, análisis por categorías, insights clave, veredicto, recomendaciones y limitaciones.
- **4 requisitos** para gráficos: títulos descriptivos con hipótesis, etiquetas en ejes, anotaciones estadísticas (medianas, porcentajes) y referencias visuales con equivalencias prácticas.
- **Nomenclatura estándar**: `planX_{hipotesis}_estadisticas.txt`, `planX_{hipotesis}_{tipo}.png`, carpeta `notebooks/results/planX-{nombre}/`.

---

## Diseño Resumido

### Módulos principales

| Módulo | Responsabilidad |
|--------|----------------|
| `notebooks/01_eda_inicial.ipynb` | Exploración inicial, estadísticas descriptivas globales |
| `notebooks/planX_*_analysis.py` | Análisis individual por hipótesis con PySpark + visualización |
| `dashboard/dashboard.py` | Presentación interactiva de resultados para stakeholders |
| `data/ai_dev_productivity.csv` | Fuente única de datos del proyecto |
| `docs/` | Documentación técnica y del pipeline |
| `plans/` | Planes estructurados por hipótesis en Markdown |

### Flujo de ejecución por plan

```
1. Iniciar sesión Spark local (local[*])
2. Cargar CSV con inferencia de schema
3. Análisis estadístico con PySpark SQL
4. Verificación y complemento con Pandas
5. Generación de visualizaciones (Matplotlib / Seaborn)
6. Guardado de estadísticas en .txt y gráficos en .png
7. Resultados disponibles en notebooks/results/{plan}/
```

---

## Riesgos Identificados

### 🔴 Críticos

| # | Riesgo | Evidencia | Impacto |
|---|--------|-----------|---------|
| R-001 | Dependencia de Java 11 frágil | PySpark requiere Java 11 específicamente; incompatible con versiones superiores o inferiores | Alto — sin Java, el análisis PySpark no ejecuta |
| R-002 | Dataset estático sin versioning | `ai_dev_productivity.csv` no tiene control de versiones ni hash de integridad | Medio-Alto — un cambio accidental invalida todos los resultados |

### 🟡 Moderados

| # | Riesgo | Evidencia | Impacto |
|---|--------|-----------|---------|
| R-003 | Sin tests automatizados | No hay archivos `test_*.py` ni pytest en el proyecto | Medio — errores en scripts no se detectan hasta ejecución manual |
| R-004 | Acoplamiento fuerte entre scripts y rutas | Scripts usan rutas relativas hardcodeadas; cambiar estructura rompe ejecución | Medio — refactoring de carpetas requiere editar todos los scripts |
| R-005 | Dashboard con imágenes estáticas | `dashboard/assets/` contiene copias de las imágenes, no referencia `notebooks/results/` | Bajo-Medio — actualizar análisis requiere sincronizar manualmente dos ubicaciones |
| R-006 | Python 3.14 en etapa beta | Algunas dependencias pueden no ser compatibles con esta versión aún | Bajo — entorno funciona pero puede generar warnings o incompatibilidades futuras |

### 🔵 Deuda técnica

| # | Deuda | Evidencia | Recomendación |
|---|-------|-----------|---------------|
| DT-001 | `requirements.txt` raíz vs `dashboard/requirements.txt` | Dos archivos de dependencias no sincronizados | Unificar o documentar diferencias explícitamente |
| DT-002 | `assets/` raíz duplica imágenes | Carpeta `./assets/` con las mismas 19 imágenes que `dashboard/assets/` | Eliminar duplicado y centralizar en `dashboard/assets/` |
| DT-003 | `.devcontainer` sin uso aparente | `devcontainer.json` presente pero no documentado en README | Documentar su uso o eliminar si no es parte del flujo activo |
| DT-004 | Licencia pendiente | README tiene `<!-- TODO: Agregar licencia -->` | Definir licencia, especialmente relevante en contexto académico |

---

## Ambigüedades sin resolver

Estas preguntas quedaron abiertas luego del escaneo y la inicialización del pipeline. Se marcan para revisión futura:

- `<!-- TODO: verificar -->` ¿El `.devcontainer` está en uso activo o es un remanente? ¿Se usa en el entorno académico?
- `<!-- TODO: verificar -->` ¿Se van a agregar nuevos planes de análisis más allá del Plan 5, o el análisis de hipótesis está cerrado?
- `<!-- TODO: verificar -->` ¿Los cambios mencionados al inicializar la documentación afectan la estructura de planes o el dashboard?
- `<!-- TODO: verificar -->` ¿El `dashboard/requirements.txt` tiene dependencias adicionales respecto al `requirements.txt` raíz?

---

## Próximos pasos recomendados

1. Definir y documentar los cambios planeados en el proyecto mediante un **change request**.
2. Resolver las ambigüedades marcadas con `<!-- TODO: verificar -->`.
3. Consolidar los dos `requirements.txt` y eliminar la carpeta `assets/` raíz duplicada.
4. Agregar licencia académica al proyecto.
5. Correr `update-docs` después de cada cambio relevante para mantener la documentación sincronizada.

---

> **Pipeline inicializado retroactivamente** — continuar con `change-request-workflow.md` para nuevas features o con `verifier-archiver-workflow.md` para validar el estado actual.