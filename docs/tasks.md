# Task Board: BigDataProject

> **Basado en:** init-pipeline (proyecto existente — retroactivo)
> **Tipo:** Proyecto académico de análisis de datos
> **Total de tasks:** 10 implementadas + 5 pendientes

---

## ✅ Features ya implementadas
> Estas features existen en el código. No hay que implementarlas.

- [x] **IMPL-000** — Configuración PySpark local sin Hadoop
  - **Estado:** implementado
  - **Archivos principales:** `notebooks/plan0_*`, `requirements.txt`
  - **Notas:** Usa `master("local[*]")`, depende de Java 11. Sin HDFS.

- [x] **IMPL-001** — EDA inicial sobre el dataset
  - **Estado:** implementado
  - **Archivos principales:** `notebooks/01_eda_inicial.ipynb`
  - **Notas:** Análisis exploratorio de las 9 variables del CSV. Punto de entrada del proyecto.

- [x] **IMPL-002** — Plan 1: Hipótesis Cafeína vs Task Success
  - **Estado:** implementado ✅ CONFIRMADA (correlación: 0.695)
  - **Archivos principales:** `notebooks/plan1_cafeina_analysis.py`, `notebooks/results/plan1-cafeina/`
  - **Notas:** 3 visualizaciones + estadísticas completas (111 líneas). Cumple estándar 100%.

- [x] **IMPL-003** — Plan 2: Hipótesis Horas de Código vs Task Success
  - **Estado:** implementado ✅ CONFIRMADA (correlación: 0.616)
  - **Archivos principales:** `notebooks/plan2_horas_codigo_analysis.py`, `notebooks/results/plan2-horas-codigo/`
  - **Notas:** 3 visualizaciones + estadísticas (121 líneas). Cumple estándar 100%.

- [x] **IMPL-004** — Plan 3: Hipótesis Carga Cognitiva vs Task Success
  - **Estado:** implementado ✅ CONFIRMADA débil (correlación: -0.200)
  - **Archivos principales:** `notebooks/plan3_cognitiva_analysis.py`, `notebooks/results/plan3-carga-cognitiva/`
  - **Notas:** 4 visualizaciones + estadísticas (118 líneas). Hallazgo relevante: sueño reduce carga (r=-0.734).

- [x] **IMPL-005** — Plan 4: Hipótesis Bugs Reportados vs Task Success
  - **Estado:** implementado ✅ CONFIRMADA muy débil (correlación: -0.178)
  - **Archivos principales:** `notebooks/plan4_bugs_analysis.py`, `notebooks/results/plan4-bugs-reportados/`
  - **Notas:** 4 visualizaciones + estadísticas (181 líneas). Umbral crítico: 4+ bugs = 0% éxito.

- [x] **IMPL-006** — Plan 5: Hipótesis Sueño vs Task Success
  - **Estado:** implementado ✅ CONFIRMADA débil (correlación: 0.187)
  - **Archivos principales:** `notebooks/plan5_sueno_analysis.py`, `notebooks/results/plan5-sueno/`
  - **Notas:** 5 visualizaciones + estadísticas (267 líneas). Punto óptimo: 7.1h con 92.3% éxito.

- [x] **IMPL-007** — Estándar de calidad obligatorio para análisis
  - **Estado:** implementado y aplicado a todos los planes
  - **Archivos principales:** `docs/SPECS.md` (sección Estándar de Calidad)
  - **Notas:** 8 secciones requeridas en estadísticas + 4 requisitos de gráficos. Regla sin excepción.

- [x] **IMPL-008** — Dashboard Streamlit con 19 visualizaciones
  - **Estado:** implementado y desplegado en Streamlit Cloud
  - **Archivos principales:** `dashboard/dashboard.py`, `dashboard/assets/` (19 imágenes), `dashboard/data/`
  - **Notas:** Incluye Dataset Explorer con filtros y análisis comparativo. Deployado en producción.

- [x] **IMPL-009** — Documentación técnica base
  - **Estado:** implementado (parcial)
  - **Archivos principales:** `docs/SPECS.md`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`, `README.md`
  - **Notas:** Documentación existente antes de correr init_doc. Ahora complementada con el pipeline completo.

---

## 📋 Tasks pendientes
> Lo que falta implementar, corregir o mejorar según el análisis.

- [ ] **TASK-001** — Definir y ejecutar cambios pendientes del proyecto <!-- TODO: verificar -->
  - **Descripción:** El usuario mencionó que hay cambios por hacer al proyecto. Pendiente de especificación.
  - **Archivos involucrados:** Por determinar
  - **Depende de:** ninguna
  - **Criterio de done:** Cambios acordados implementados y documentados en CHANGELOG.md

- [ ] **TASK-002** — Agregar licencia al proyecto
  - **Descripción:** El README tiene un `<!-- TODO: Agregar licencia -->` sin resolver. Para un proyecto académico suele ser MIT o CC.
  - **Archivos involucrados:** `README.md`, nuevo archivo `LICENSE`
  - **Depende de:** ninguna
  - **Criterio de done:** Sección de licencia completa en README + archivo LICENSE en raíz

- [ ] **TASK-003** — Configurar Git con remote correcto
  - **Descripción:** SPECS.md indica "Sin configuración Git detectada". El README menciona un remote de GitHub genérico. Para proyecto académico es importante tenerlo configurado.
  - **Archivos involucrados:** configuración git del repositorio
  - **Depende de:** ninguna
  - **Criterio de done:** `git remote -v` muestra el remote correcto y el proyecto está pusheado

- [ ] **TASK-004** — Actualizar ARCHITECTURE.md para reflejar el dashboard
  - **Descripción:** El diagrama de componentes en ARCHITECTURE.md no incluye el módulo Dashboard/Streamlit que fue agregado posteriormente.
  - **Archivos involucrados:** `docs/ARCHITECTURE.md`
  - **Depende de:** ninguna
  - **Criterio de done:** Diagrama Mermaid actualizado con el componente Streamlit y su flujo de datos

- [ ] **TASK-005** — Unificar dependencias entre requirements.txt raíz y dashboard/requirements.txt
  - **Descripción:** Existen dos archivos `requirements.txt` (raíz y `dashboard/`). No está claro si son consistentes o si el de raíz ya incluye Streamlit.
  - **Archivos involucrados:** `requirements.txt`, `dashboard/requirements.txt`
  - **Depende de:** ninguna
  - **Criterio de done:** Un solo requirements.txt con todas las dependencias, o documentación clara de para qué sirve cada uno

---

---

## 🔄 Tasks de Change Requests activos

### CR-001 — Plan 6: Uso de IA vs Task Success

- [x] **TASK-CR001-1** — Crear `notebooks/plan6_uso_ia_analysis.py`
  - **Estado:** ✅ Script creado (32 KB)
  - **Archivos:** `notebooks/plan6_uso_ia_analysis.py`
  - **Criterio de done:** Script ejecuta sin errores, genera 4 gráficos + estadísticas en `notebooks/results/plan6-uso-ia/`

- [ ] **TASK-CR001-2** — Ejecutar y validar Plan 6
  - **Descripción:** Correr el script y verificar que los resultados cumplen el estándar de calidad (8 secciones + 4 requisitos de gráficos)
  - **Archivos:** `notebooks/results/plan6-uso-ia/`
  - **Depende de:** TASK-CR001-1
  - **Criterio de done:** 4 PNGs + estadísticas.txt generados, veredicto de hipótesis determinado

- [ ] **TASK-CR001-3** — Integrar Plan 6 al dashboard
  - **Descripción:** Agregar Plan 6 a la navegación del dashboard, sección Análisis por Plan y Comparativo
  - **Archivos:** `dashboard/dashboard.py`, `dashboard/assets/`
  - **Depende de:** TASK-CR001-2
  - **Criterio de done:** Plan 6 visible en Streamlit con galería y estadísticas

---

### CR-002 — Plan 7: Trade-off Commits vs Bugs

- [x] **TASK-CR002-1** — Crear `notebooks/plan7_commits_bugs_analysis.py`
  - **Estado:** ✅ Script creado (41 KB)
  - **Archivos:** `notebooks/plan7_commits_bugs_analysis.py`
  - **Criterio de done:** Script ejecuta sin errores, genera 4 gráficos + estadísticas en `notebooks/results/plan7-commits-bugs/`

- [ ] **TASK-CR002-2** — Ejecutar y validar Plan 7
  - **Descripción:** Correr el script y verificar el trade-off. Validar rangos de commits y bugs contra distribución real del dataset
  - **Archivos:** `notebooks/results/plan7-commits-bugs/`
  - **Depende de:** TASK-CR002-1
  - **Criterio de done:** 4 PNGs + estadísticas.txt generados, trade-off confirmado o refutado con evidencia

- [ ] **TASK-CR002-3** — Integrar Plan 7 al dashboard
  - **Descripción:** Agregar Plan 7 a la navegación y sección comparativa del dashboard
  - **Archivos:** `dashboard/dashboard.py`, `dashboard/assets/`
  - **Depende de:** TASK-CR002-2
  - **Criterio de done:** Plan 7 visible en Streamlit con galería y estadísticas

---

### CR-003 — Plan 8: Balance Óptimo Multivariado

- [x] **TASK-CR003-1** — Crear `notebooks/plan8_balance_optimo_analysis.py`
  - **Estado:** ✅ Script creado (35 KB)
  - **Archivos:** `notebooks/plan8_balance_optimo_analysis.py`
  - **Criterio de done:** Script ejecuta sin errores, genera 5 gráficos + estadísticas en `notebooks/results/plan8-balance-optimo/`

- [ ] **TASK-CR003-2** — Ejecutar y validar Plan 8
  - **Descripción:** Correr el script y verificar que los heatmaps, scatter de zona dorada y score compuesto son correctos y cumplen el estándar
  - **Archivos:** `notebooks/results/plan8-balance-optimo/`
  - **Depende de:** TASK-CR003-1
  - **Criterio de done:** 5 PNGs + estadísticas.txt generados, zona dorada identificada con evidencia cuantitativa

- [ ] **TASK-CR003-3** — Integrar Plan 8 al dashboard
  - **Descripción:** Agregar Plan 8 al dashboard como sección especial de análisis multivariado
  - **Archivos:** `dashboard/dashboard.py`, `dashboard/assets/`
  - **Depende de:** TASK-CR003-2
  - **Criterio de done:** Plan 8 visible en Streamlit, destacado como análisis avanzado

---

## Orden de ejecución sugerido

```
TASK-CR001-2  (ejecutar Plan 6)
TASK-CR002-2  (ejecutar Plan 7)   ← paralelo con CR001-2 y CR003-2
TASK-CR003-2  (ejecutar Plan 8)

  → TASK-CR001-3 + TASK-CR002-3 + TASK-CR003-3  (integrar los 3 al dashboard juntos)

  → TASK-002 (licencia)
  → TASK-003 (Git)
  → TASK-004 (actualizar arquitectura)
  → TASK-005 (unificar dependencias)
```

---

> **Siguiente paso recomendado:** Ejecutar los 3 scripts nuevos (Planes 6, 7 y 8) y verificar
> que generan resultados correctos antes de integrarlos al dashboard.
> Correr `update-docs` después de cada ejecución exitosa.