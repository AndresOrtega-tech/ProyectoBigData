# CR-001: Plan 6 — Análisis Uso de IA vs Task Success

> **Tipo de cambio:** MEDIUM
> **Estado:** En implementación
> **Fecha:** 2025
> **Confirmado por:** Andres

---

## ¿Qué cambia?

Se agrega el **Plan 6** al proyecto: análisis de la variable `ai_usage_hours` contra `task_success`.

Esta variable existe en el dataset desde el inicio pero nunca fue analizada como hipótesis independiente. Es la única variable del dataset sin análisis propio, y es especialmente relevante dado que el proyecto estudia la productividad de desarrolladores que usan herramientas de IA.

---

## ¿Por qué?

Para responder la pregunta de investigación:

> **"¿El uso de IA aumenta o disminuye la probabilidad de éxito en tareas de desarrollo?"**

Sin este análisis, el ranking de correlaciones del dashboard está incompleto y la variable más representativa del tema central del proyecto queda sin evidencia cuantitativa.

---

## Hipótesis a validar

> "Mayor uso de herramientas de IA está asociado con mayor task_success"

- **Variable independiente:** `ai_usage_hours`
- **Variable dependiente:** `task_success` (0 = fallo, 1 = éxito)
- **Correlación esperada:** a determinar con el análisis

---

## Clasificación de impacto

**MEDIUM** — Agrega un módulo de análisis nuevo sin modificar los existentes. Requiere:
- Nuevo script de análisis
- Nueva carpeta de resultados
- Actualización del dashboard (en CR posterior)
- Actualización de docs

---

## Archivos afectados

| Archivo | Acción |
|---------|--------|
| `notebooks/plan6_uso_ia_analysis.py` | Crear (nuevo script de análisis) |
| `notebooks/results/plan6-uso-ia/` | Crear (carpeta de resultados) |
| `docs/tasks.md` | Actualizar con TASK-006 |
| `docs/SPECS.md` | Actualizar con Plan 6 |
| `docs/CHANGELOG.md` | Registrar cambio |
| `dashboard/dashboard.py` | Actualizar en CR-004 (pendiente) |
| `dashboard/assets/` | Agregar imágenes en CR-004 |

---

## Salidas esperadas del script

### Visualizaciones (mínimo 4 gráficos)

| Archivo | Descripción |
|---------|-------------|
| `plan6_uso_ia_boxplot.png` | Distribución de ai_usage_hours por task_success con medianas anotadas |
| `plan6_uso_ia_histograma.png` | Distribución con rangos de uso visibles |
| `plan6_uso_ia_tasa_exito.png` | Tasa de éxito por rango de horas de IA |
| `plan6_uso_ia_scatter.png` | Dispersión ai_usage_hours vs hours_coding coloreado por task_success |

### Archivo de estadísticas

`plan6_uso_ia_estadisticas.txt` — debe cumplir las **8 secciones obligatorias**:
1. Metodología
2. Definición de rangos
3. Estadísticas descriptivas
4. Análisis por categorías
5. Insights clave
6. Veredicto
7. Recomendaciones prácticas
8. Limitaciones del análisis

---

## Rangos sugeridos para ai_usage_hours

A confirmar con la distribución real del dataset, pero como punto de partida:

| Rango | Umbral | Equivalencia |
|-------|--------|--------------|
| Bajo | < 2h | Uso esporádico de IA |
| Medio | 2–4h | Uso moderado de IA |
| Alto | > 4h | Uso intensivo de IA |

---

## Criterio de done

- [ ] Script ejecuta sin errores desde la raíz del proyecto
- [ ] 4 visualizaciones generadas en `notebooks/results/plan6-uso-ia/`
- [ ] Archivo de estadísticas con 8 secciones completo
- [ ] Veredicto de hipótesis determinado con correlación de Pearson
- [ ] Cumple estándar de calidad obligatorio (8 secciones + 4 requisitos de gráficos)
- [ ] Listo para integrar al dashboard en CR-004

---

## Tasks generadas

Ver `docs/tasks.md` → **TASK-006**