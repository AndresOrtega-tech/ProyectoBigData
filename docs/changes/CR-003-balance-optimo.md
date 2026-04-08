# CR-003 — Análisis Multivariado: Balance Óptimo Cafeína + Horas + Sueño

> **ID:** CR-003
> **Nombre:** balance-optimo
> **Fecha:** 2025
> **Estado:** 🟡 En implementación
> **Impacto:** LARGE
> **Solicitado por:** Andres

---

## ¿Qué cambia?

Agregar un nuevo script de análisis multivariado que estudia la combinación óptima de tres variables
(`coffee_intake_mg`, `hours_coding`, `sleep_hours`) en relación con `task_success`.

A diferencia de los planes anteriores (que analizan una variable a la vez), este plan busca identificar
la "zona dorada" donde la combinación de los tres factores maximiza la tasa de éxito.

---

## ¿Por qué?

Los análisis individuales (Planes 1, 2 y 5) confirman que cafeína, horas de código y sueño tienen
correlación positiva con el éxito. Sin embargo, no responden: **¿qué combinación de las tres
maximiza la probabilidad de éxito?**

Un desarrollador podría dormir 7h, tomar 400mg de cafeína y trabajar 8h — pero ¿es esa la
combinación óptima o hay una mejor? Este plan responde esa pregunta con evidencia del dataset.

---

## Clasificación de impacto

**LARGE** — requiere:
- Script de análisis nuevo con lógica multivariada
- Múltiples visualizaciones de pares de variables (heatmaps 2D)
- Archivo de estadísticas con análisis cruzado
- Nueva carpeta de resultados
- Actualización del dashboard (en CR separado o como parte del cierre)

---

## Hipótesis

> "Existe un balance óptimo entre cafeína, horas de código y sueño que maximiza significativamente
> la tasa de task_success, superior al efecto individual de cada variable."

---

## Variables involucradas

| Variable | Rol | Planes que la analizan individualmente |
|----------|-----|----------------------------------------|
| `coffee_intake_mg` | Predictor 1 | Plan 1 (r=0.695) |
| `hours_coding` | Predictor 2 | Plan 2 (r=0.616) |
| `sleep_hours` | Predictor 3 | Plan 5 (r=0.187) |
| `task_success` | Variable objetivo | Todos los planes |

---

## Análisis a realizar

### 1. Score compuesto
Crear un score ponderado por correlación para cada registro:
```
score = (0.695 × cafeina_norm) + (0.616 × horas_norm) + (0.187 × sueno_norm)
```
Analizar la correlación del score compuesto vs `task_success`.

### 2. Heatmaps de pares (3 combinaciones)
- **Cafeína × Horas** → tasa de éxito por celda
- **Cafeína × Sueño** → tasa de éxito por celda
- **Horas × Sueño** → tasa de éxito por celda

### 3. Zona dorada
Identificar el rango de cada variable donde la combinación produce la máxima tasa de éxito.
Visualizar con scatter plot 2D proyectado (cafeína + horas), coloreado por task_success.

### 4. Ranking de combinaciones
Top 5 combinaciones de rangos con mayor tasa de éxito, ordenadas descendentemente.

---

## Archivos a crear

| Archivo | Descripción |
|---------|-------------|
| `notebooks/plan8_balance_optimo_analysis.py` | Script principal del análisis |
| `notebooks/results/plan8-balance-optimo/plan8_heatmap_cafeina_horas.png` | Heatmap cafeína × horas |
| `notebooks/results/plan8-balance-optimo/plan8_heatmap_cafeina_sueno.png` | Heatmap cafeína × sueño |
| `notebooks/results/plan8-balance-optimo/plan8_heatmap_horas_sueno.png` | Heatmap horas × sueño |
| `notebooks/results/plan8-balance-optimo/plan8_scatter_zona_dorada.png` | Scatter zona dorada |
| `notebooks/results/plan8-balance-optimo/plan8_score_compuesto.png` | Score compuesto vs éxito |
| `notebooks/results/plan8-balance-optimo/plan8_balance_estadisticas.txt` | Estadísticas completas |

---

## Archivos afectados / a actualizar

| Archivo | Cambio |
|---------|--------|
| `docs/tasks.md` | Agregar TASK-006 |
| `docs/CHANGELOG.md` | Registrar CR-003 |
| `docs/spec.md` | Agregar sección 4.9 Plan 8 |
| `docs/design.md` | Actualizar tabla de componentes y resultados |
| `dashboard/dashboard.py` | Agregar Plan 8 a navegación y sección Comparativa (CR posterior) |

---

## Estándar de calidad obligatorio

El script debe cumplir el estándar sin excepciones:

**Estadísticas (8 secciones):**
1. Metodología clara (variables, técnicas, ponderación usada)
2. Definición de rangos para las 3 variables con equivalencias prácticas
3. Estadísticas descriptivas del score compuesto (promedio, mediana, min/max, desviación)
4. Análisis por combinaciones de rangos con tasa de éxito
5. Insights clave explicados con sustento cuantitativo
6. Veredicto: ¿existe una zona dorada estadísticamente significativa?
7. Recomendaciones prácticas: combinación óptima sugerida
8. Limitaciones del análisis multivariado

**Gráficos (4 requisitos mínimos por gráfico):**
1. Títulos descriptivos con la hipótesis
2. Etiquetas claras en ambos ejes
3. Anotaciones estadísticas (tasas de éxito, zonas resaltadas)
4. Referencias visuales y equivalencias prácticas

---

## Tasks nuevas en tasks.md

```
- [ ] TASK-006 — Implementar Plan 8: análisis multivariado balance óptimo
  - Archivos: notebooks/plan8_balance_optimo_analysis.py,
              notebooks/results/plan8-balance-optimo/
  - Depende de: ninguna (datos ya disponibles)
  - Criterio de done: script ejecuta sin errores, genera 6 archivos de salida,
                      cumple estándar de 8 secciones + 4 requisitos de gráficos
```

---

## Criterio de done del CR

- [ ] Script `plan8_balance_optimo_analysis.py` ejecuta sin errores
- [ ] Se generan los 6 archivos de salida (5 gráficos + 1 estadísticas)
- [ ] El archivo de estadísticas tiene las 8 secciones requeridas
- [ ] Cada gráfico cumple los 4 requisitos mínimos
- [ ] `docs/CHANGELOG.md` actualizado
- [ ] Dashboard actualizado con Plan 8 (puede ser en CR posterior)

---

> **Siguiente paso recomendado:** Correr `update-docs` al completar la implementación.