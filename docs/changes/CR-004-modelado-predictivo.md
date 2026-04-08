# CR-004 — Fases III, IV y V: Preparación, Modelado y Evaluación con Predictor Interactivo

> **ID:** CR-004
> **Nombre:** modelado-predictivo
> **Clasificación:** LARGE
> **Estado:** 🔄 En implementación
> **Fecha:** 2025
> **Solicitado por:** Andres

---

## ¿Qué cambia?

Se implementan las tres fases finales del proyecto académico:

- **FASE III** — Preparación de datos: limpieza, feature engineering, split y balance de clases
- **FASE IV** — Modelado: Logistic Regression, Decision Tree y Random Forest con optimización de hiperparámetros
- **FASE V** — Evaluación: métricas completas, análisis de errores, feature importance e insights accionables

Adicionalmente, se agrega al dashboard una **nueva sección de Predictor Interactivo** donde el usuario puede ingresar sus propios valores y obtener una predicción de `task_success` en tiempo real, con explicación del modelo y probabilidad de éxito.

---

## ¿Por qué?

Las fases anteriores (I-II) analizaron correlaciones individuales e hipótesis. Esta fase cierra el ciclo del proyecto:

1. Construye un modelo predictivo real basado en la evidencia recolectada
2. Cuantifica la importancia relativa de cada variable con el modelo entrenado
3. Genera una herramienta usable: el predictor convierte el análisis estadístico en algo accionable para cualquier desarrollador
4. Responde la pregunta central del proyecto: ¿podemos predecir el éxito de una tarea?

---

## Clasificación de impacto

**LARGE** — involucra:
- 2 scripts Python nuevos con lógica compleja (preparación + modelado)
- Instalación de 3 dependencias nuevas (scikit-learn, imbalanced-learn, joblib)
- Artefactos persistentes que el dashboard consume en runtime (modelo serializado)
- Actualización significativa del dashboard con nueva sección interactiva
- Múltiples visualizaciones nuevas

---

## Dependencias instaladas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `scikit-learn` | 1.8.0 | Modelos ML, métricas, preprocessing, GridSearchCV |
| `imbalanced-learn` | 0.14.1 | SMOTE para balance de clases si es necesario |
| `joblib` | 1.5.3 | Serialización del modelo entrenado |

---

## Arquitectura de los scripts

### Script A: `notebooks/fase3_preparacion_datos.py`

**Responsabilidad:** Limpiar, enriquecer y dividir el dataset.

#### Pasos:
1. **Carga del dataset** — `data/ai_dev_productivity.csv`
2. **Limpieza:**
   - Detectar y reportar valores nulos
   - Detectar outliers con IQR (no eliminar automáticamente — reportar y decidir)
   - Validar consistencias lógicas (ej: hours_coding > 24)
3. **Feature Engineering** — nuevas variables derivadas:
   - `sleep_deficit` = `8 - sleep_hours` (positivo = déficit, negativo = exceso)
   - `productivity_ratio` = `commits / hours_coding` (commits por hora, div segura)
   - `caffeine_per_hour` = `coffee_intake_mg / hours_coding` (intensidad por hora)
   - `work_intensity` = `hours_coding × cognitive_load`
   - `coffee_category_num`: bajo=0, medio=1, alto=2 (basado en <200 / 200-400 / >400)
   - `sleep_category_num`: insuficiente=0, óptimo=1, excesivo=2 (basado en <6 / 6-8 / >8)
4. **Balance de clases** — verificar proporción de `task_success`. Si desbalance > 30%, aplicar SMOTE solo sobre training set.
5. **Split estratificado** — 70% train / 15% val / 15% test, `random_state=42`, `stratify=y`
6. **Guardado de artefactos:**
   - `notebooks/results/fase3-preparacion/X_train.csv`
   - `notebooks/results/fase3-preparacion/X_val.csv`
   - `notebooks/results/fase3-preparacion/X_test.csv`
   - `notebooks/results/fase3-preparacion/y_train.csv`
   - `notebooks/results/fase3-preparacion/y_val.csv`
   - `notebooks/results/fase3-preparacion/y_test.csv`
   - `notebooks/results/fase3-preparacion/feature_names.txt`
   - `notebooks/results/fase3-preparacion/fase3_estadisticas.txt`
7. **Visualizaciones:**
   - `fase3_distribucion_clases.png` — balance antes/después
   - `fase3_outliers.png` — boxplots de todas las variables
   - `fase3_correlation_matrix.png` — heatmap de correlación con features nuevas
   - `fase3_feature_engineering.png` — distribución de las 4 variables nuevas

---

### Script B: `notebooks/fase45_modelado_evaluacion.py`

**Responsabilidad:** Entrenar, optimizar, evaluar y guardar el modelo final.

#### Pasos:
1. **Carga de datos** — desde `notebooks/results/fase3-preparacion/`
2. **Entrenamiento de 3 modelos base** (sin tuning):
   - `LogisticRegression` (baseline)
   - `DecisionTreeClassifier`
   - `RandomForestClassifier`
3. **Comparación inicial** en validation set (accuracy, F1, AUC-ROC)
4. **Optimización** del mejor modelo con `GridSearchCV` (5-fold CV sobre training)
5. **Evaluación final** del modelo optimizado en test set (datos nunca vistos):
   - Accuracy, Precision, Recall, F1-Score
   - AUC-ROC con curva
   - Confusion Matrix
   - Reporte completo por clase
6. **Feature Importance** — top variables del modelo final
7. **Análisis de errores** — patrones en falsos positivos y falsos negativos
8. **Guardado del modelo** para el predictor:
   - `dashboard/model/best_model.pkl` — pipeline completo (scaler + modelo)
   - `dashboard/model/feature_names.pkl` — lista de features en orden
   - `dashboard/model/model_metadata.json` — métricas finales, nombre del modelo, params
9. **Visualizaciones:**
   - `fase45_model_comparison.png` — comparación de los 3 modelos
   - `fase45_confusion_matrix.png` — matriz de confusión del modelo final
   - `fase45_roc_curve.png` — curva ROC con AUC
   - `fase45_feature_importance.png` — importancia de cada variable
   - `fase45_learning_curve.png` — curva de aprendizaje (overfitting check)
10. **Archivo de estadísticas** — `fase45_estadisticas.txt` con las 8 secciones

---

## Contrato de features (Training ↔ Predictor)

El conjunto de features que entra al modelo es EXACTAMENTE este, en este orden:

```
hours_coding, coffee_intake_mg, distractions, sleep_hours, commits,
bugs_reported, ai_usage_hours, cognitive_load,
sleep_deficit, productivity_ratio, caffeine_per_hour, work_intensity,
coffee_category_num, sleep_category_num
```

**Total: 14 features**

El dashboard predictor recibe los 8 valores originales del usuario, calcula las 6 features derivadas con la misma fórmula, y pasa las 14 al modelo en ese orden.

---

## Dashboard — Sección Predictor Interactivo

### Ubicación
Nueva opción en el sidebar: `"🤖 Predictor de Éxito"`

### Contenido de la sección

#### Bloque 1: Explicación del modelo
- Qué hace, qué predice, cómo fue entrenado
- Métricas reales del modelo (cargadas desde `model_metadata.json`)
- Advertencia sobre correlación vs causalidad

#### Bloque 2: Inputs del usuario
Sliders con descripción de cada variable:

| Variable | Label en UI | Rango | Descripción |
|----------|-------------|-------|-------------|
| `hours_coding` | "Horas de código hoy" | 0–16 | Horas programando en la sesión |
| `coffee_intake_mg` | "Cafeína consumida (mg)" | 0–800 | 1 taza ≈ 95mg |
| `sleep_hours` | "Horas de sueño anoche" | 3–12 | Horas dormidas la noche anterior |
| `cognitive_load` | "Carga cognitiva (1-10)" | 1–10 | Qué tan compleja sentís la tarea |
| `distractions` | "Número de distracciones" | 0–15 | Interrupciones durante el trabajo |
| `commits` | "Commits realizados" | 0–20 | Commits en la sesión |
| `bugs_reported` | "Bugs encontrados" | 0–10 | Bugs detectados en la sesión |
| `ai_usage_hours` | "Horas usando herramientas IA" | 0–8 | Copilot, ChatGPT, etc. |

#### Bloque 3: Features derivadas (calculadas automáticamente, mostradas al usuario)
- Mostrar los valores calculados como métricas informativas: sleep_deficit, productivity_ratio, etc.
- Esto educa al usuario sobre qué está mirando el modelo

#### Bloque 4: Resultado de la predicción
- `st.metric()` grande con la predicción (✅ ÉXITO o ❌ FRACASO)
- Barra de probabilidad (gauge o progress bar) con % de probabilidad de éxito
- Interpretación contextualizada: "Tu combinación de factores sugiere..."
- Top 3 factores que más influyeron en esta predicción específica (basado en feature importance × valor del usuario)

---

## Archivos a crear / modificar

| Archivo | Acción |
|---------|--------|
| `notebooks/fase3_preparacion_datos.py` | Crear |
| `notebooks/fase45_modelado_evaluacion.py` | Crear |
| `notebooks/results/fase3-preparacion/` | Crear (directorio) |
| `notebooks/results/fase45-modelado/` | Crear (directorio) |
| `dashboard/model/` | Crear (directorio) |
| `dashboard/model/best_model.pkl` | Generado al ejecutar fase45 |
| `dashboard/model/feature_names.pkl` | Generado al ejecutar fase45 |
| `dashboard/model/model_metadata.json` | Generado al ejecutar fase45 |
| `dashboard/assets/fase45_*.png` | Copiar desde results al ejecutar |
| `dashboard/dashboard.py` | Agregar sección Predictor |
| `requirements.txt` | Agregar scikit-learn, imbalanced-learn, joblib |
| `docs/tasks.md` | Agregar tasks CR-004 |
| `docs/CHANGELOG.md` | Registrar CR-004 |

---

## Orden de ejecución (importante)

```
1. python3 notebooks/fase3_preparacion_datos.py
   → Genera los CSVs de train/val/test en notebooks/results/fase3-preparacion/

2. python3 notebooks/fase45_modelado_evaluacion.py
   → Lee los CSVs, entrena modelos, guarda dashboard/model/*.pkl y .json

3. Copiar imágenes de fase45 a dashboard/assets/
   cp notebooks/results/fase45-modelado/*.png dashboard/assets/

4. Streamlit recarga automáticamente el predictor
```

---

## Criterio de done

- [ ] `fase3_preparacion_datos.py` ejecuta sin errores
- [ ] CSVs de split generados correctamente con estratificación
- [ ] Feature engineering produce las 6 variables derivadas sin NaN ni Inf
- [ ] `fase45_modelado_evaluacion.py` ejecuta sin errores
- [ ] Modelo final supera 80% accuracy en test set (objetivo académico)
- [ ] `best_model.pkl`, `feature_names.pkl` y `model_metadata.json` guardados en `dashboard/model/`
- [ ] 5 visualizaciones de fase45 generadas en `notebooks/results/fase45-modelado/`
- [ ] Predictor en Streamlit funciona: recibe inputs, predice, muestra probabilidad
- [ ] `requirements.txt` actualizado con las 3 nuevas dependencias
- [ ] `docs/tasks.md` actualizado con tasks de este CR

---

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Accuracy < 80% con los 3 modelos | Media | Probar XGBoost como fallback; ajustar features |
| Dataset desbalanceado requiere SMOTE | Baja (60/40 aprox) | Script detecta automáticamente y aplica si ratio > 1.3 |
| Overfitting en Decision Tree | Alta | Limitar `max_depth` en GridSearch |
| Features derivadas con división por cero | Media | División segura con `np.where` y `fillna(0)` |
| Modelo incompatible entre training y predictor | Baja | Contrato de features explícito en este CR |

---

> **Siguiente paso:** Ejecutar `fase3_preparacion_datos.py` → `fase45_modelado_evaluacion.py` → actualizar dashboard con sección predictor.
> Correr `update-docs` al finalizar para mantener CHANGELOG y docs sincronizados.