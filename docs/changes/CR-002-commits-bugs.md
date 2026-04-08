# CR-002 — Trade-off entre Commits y Bugs Reportados

> **ID:** CR-002
> **Nombre:** commits-bugs-tradeoff
> **Clasificación:** MEDIUM
> **Estado:** pendiente de implementación
> **Fecha:** 2025
> **Solicitado por:** Andres

---

## ¿Qué cambia?

Agregar un nuevo plan de análisis (Plan 7) que estudie la relación entre la cantidad de commits (`commits`) y la calidad del código medida como bugs reportados (`bugs_reported`), evaluando si existe un trade-off entre productividad y calidad, y cómo ambas variables combinadas afectan a `task_success`.

---

## ¿Por qué?

La pregunta "¿Hay trade-offs entre cantidad (commits) y calidad (bugs)?" no está respondida en el proyecto actual. `commits` es una variable del dataset que nunca fue analizada de forma independiente. El Plan 4 analiza `bugs_reported` de forma aislada, pero no explora su relación con `commits` ni el impacto combinado de ambas sobre el éxito de las tareas.

---

## Hipótesis a validar

> "A mayor número de commits, mayor probabilidad de que también aumenten los bugs reportados, generando un trade-off entre cantidad y calidad que afecta negativamente a `task_success`."

---

## Alcance del cambio

### Archivos nuevos
- `notebooks/plan7_commits_bugs_analysis.py` — script principal de análisis
- `notebooks/results/plan7-commits-bugs/plan7_commits_bugs_scatter.png`
- `notebooks/results/plan7-commits-bugs/plan7_commits_bugs_tasa_exito_commits.png`
- `notebooks/results/plan7-commits-bugs/plan7_commits_bugs_tasa_exito_bugs.png`
- `notebooks/results/plan7-commits-bugs/plan7_commits_bugs_heatmap.png`
- `notebooks/results/plan7-commits-bugs/plan7_commits_bugs_estadisticas.txt`

### Archivos afectados (actualización posterior)
- `docs/tasks.md` — marcar TASK-001 como avanzando, agregar tasks de este CR
- `docs/CHANGELOG.md` — registrar nuevo plan
- `dashboard/` — pendiente de integración en CR separado

---

## Clasificación de impacto

**MEDIUM** — Agrega un módulo nuevo sin modificar los existentes. No toca el dashboard, el EDA ni los planes anteriores. El único riesgo es de nomenclatura y consistencia con el estándar de calidad.

---

## Análisis a realizar

| Análisis | Herramienta | Visualización |
|----------|-------------|---------------|
| Correlación commits vs task_success | PySpark `stat.corr` | — |
| Correlación bugs vs task_success | PySpark `stat.corr` | — |
| Scatter: commits vs bugs coloreado por task_success | Pandas + Matplotlib | `plan7_commits_bugs_scatter.png` |
| Tasa de éxito por rango de commits | PySpark groupBy + Pandas | `plan7_commits_bugs_tasa_exito_commits.png` |
| Tasa de éxito por rango de bugs | PySpark groupBy + Pandas | `plan7_commits_bugs_tasa_exito_bugs.png` |
| Heatmap: rango_commits × rango_bugs → tasa éxito | Pandas pivot + Seaborn | `plan7_commits_bugs_heatmap.png` |

---

## Rangos sugeridos

**Commits:**
- Bajo: < 3 commits
- Medio: 3–6 commits
- Alto: > 6 commits

**Bugs reportados:**
- Sin bugs: 0
- Pocos: 1–2
- Crítico: 3+

> Los rangos deben validarse contra la distribución real del dataset antes de aplicarlos.

---

## Estándar de calidad obligatorio

El archivo de estadísticas debe tener las 8 secciones requeridas:
1. Metodología
2. Definición de rangos
3. Estadísticas descriptivas
4. Análisis por categorías
5. Insights clave
6. Veredicto
7. Recomendaciones
8. Limitaciones

Los gráficos deben cumplir los 4 requisitos:
1. Título descriptivo con hipótesis
2. Etiquetas en ambos ejes
3. Anotaciones estadísticas (medianas, porcentajes, conteos)
4. Referencias visuales con equivalencias prácticas

---

## Tasks nuevas para docs/tasks.md

- [ ] **TASK-CR002-1** — Crear `notebooks/plan7_commits_bugs_analysis.py` siguiendo el patrón de planes anteriores
  - **Criterio de done:** Script ejecuta sin errores, genera 4 gráficos y estadísticas con estándar 100%

- [ ] **TASK-CR002-2** — Verificar rangos óptimos de commits y bugs contra distribución real del dataset
  - **Criterio de done:** Rangos definidos con justificación estadística en el archivo de estadísticas

- [ ] **TASK-CR002-3** — Integrar resultados en el dashboard (diferido — CR de dashboard)
  - **Criterio de done:** Plan 7 visible en dashboard con galería y estadísticas

---

## Preguntas abiertas

- `<!-- TODO: verificar -->` ¿Se esperaba que el scatter muestre alta correlación positiva commits↔bugs, o son independientes?
- `<!-- TODO: verificar -->` ¿El análisis debe incluir `commits` vs `task_success` como hipótesis independiente o solo en contexto del trade-off con bugs?

---

## Recomendación post-implementación

Correr `update-docs` después de completar el script para mantener `CHANGELOG.md` y `ARCHITECTURE.md` actualizados con el nuevo plan.