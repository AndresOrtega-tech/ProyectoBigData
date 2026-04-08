import json as json_lib
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="BigData Project - Dashboard de Productividad",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title("🚀 BigData Project - Dashboard de Productividad de Desarrolladores")
st.markdown("### Análisis de 8 hipótesis con PySpark + Streamlit")
st.markdown("---")


# Helper: convierte clave interna (Plan X) a etiqueta visible (Hipótesis X)
def display_label(key: str) -> str:
    if key.startswith("Plan "):
        return key.replace("Plan ", "Hipótesis ")
    return key


# Datos de las hipótesis
# Nota: Plan 1-5 mantienen sus claves originales porque se usan en rutas de imágenes.
# Las nuevas hipótesis usan claves "Hipótesis 6/7/8" directamente.
hipotesis_data = {
    "Plan 1": {
        "nombre": "Cafeína vs Éxito",
        "correlacion": 0.695,
        "veredicto": "✅ CONFIRMADA",
        "color": "green",
        "insights": ">400mg = 83.6% éxito",
        "imagenes": [
            "plan1_cafeina_boxplot.png",
            "plan1_cafeina_histograma.png",
            "plan1_cafeina_tasa_exito.png",
        ],
        "descripcion": [
            "Distribución con anotaciones",
            "Patrones con rangos visibles",
            "Relación dose-respuesta",
        ],
        "resultados": """
**Correlación de Pearson:** +0.695 (Fuerte positiva)
**Variable:** coffee_intake_mg vs task_success

**Estadísticas por grupo:**
- Grupo Éxito: promedio 461.4 mg, mediana 457 mg
- Grupo Fracaso: promedio 202.6 mg, mediana 190 mg
- Diferencia: +258.8 mg más en el grupo exitoso (+127.7%)

**Tasa de éxito por rango:**
- Bajo (<200mg / ~2 tazas): **0% éxito** ← umbral crítico
- Medio (200-400mg / 2-4 tazas): **9.8% éxito**
- Alto (>400mg / >4 tazas): **83.6% éxito**

**Insights clave:**
- Consumir >400mg aumenta **8.5x** la probabilidad de éxito vs rango medio
- El grupo exitoso tiene **menor variabilidad** (std=67mg vs 140mg): consumo más consistente
- La relación es dose-respuesta: a mayor dosis, mayor éxito

**Recomendación:** Mantener consumo >400mg (>4 tazas) para máxima probabilidad de éxito
""",
    },
    "Plan 2": {
        "nombre": "Horas de Código vs Éxito",
        "correlacion": 0.616,
        "veredicto": "✅ CONFIRMADA",
        "color": "green",
        "insights": "6-9h = 85.7% éxito",
        "imagenes": [
            "plan2_horas_boxplot.png",
            "plan2_horas_histograma.png",
            "plan2_horas_tasa_exito.png",
        ],
        "descripcion": [
            "Distribución con medianas",
            "Patrones con líneas de referencia",
            "Tasa de éxito por rango",
        ],
        "resultados": """
**Correlación de Pearson:** +0.616 (Fuerte positiva)
**Variable:** hours_coding vs task_success

**Estadísticas por grupo:**
- Grupo Éxito: promedio 5.98h, mediana 5.8h
- Grupo Fracaso: promedio 3.53h, mediana 3.4h
- Diferencia: +2.45h más en el grupo exitoso (+69.4%)

**Tasa de éxito por rango:**
- Bajo (<3h): **0% éxito** ← umbral crítico
- Medio (3-6h): tasa moderada
- Óptimo (6-9h): **85.7% éxito**
- Exceso (>9h): **76.9% éxito** ← rendimientos decrecientes

**Insights clave:**
- Zona muerta: menos de 3h de código garantiza fracaso en este dataset
- La zona óptima es 6-9h: más no siempre es mejor
- Rendimientos decrecientes a partir de 9h (sobrecarga, fatiga)

**Recomendación:** Sesiones de 6-9h de código para maximizar probabilidad de éxito
""",
    },
    "Plan 3": {
        "nombre": "Carga Cognitiva vs Éxito",
        "correlacion": -0.200,
        "veredicto": "✅ CONFIRMADA (débil)",
        "color": "orange",
        "insights": "Alta carga = 50.3% éxito",
        "imagenes": [
            "plan3_cognitiva_boxplot.png",
            "plan3_cognitiva_scatter.png",
            "plan3_cognitiva_heatmap.png",
            "plan3_cognitiva_tasa_exito.png",
        ],
        "descripcion": [
            "Distribución con medianas",
            "Interacción carga vs horas",
            "Tasa de éxito por combinación",
            "Tasa de éxito por nivel",
        ],
        "resultados": """
**Correlación de Pearson:** -0.200 (Débil negativa)
**Variable:** cognitive_load vs task_success

**Estadísticas por grupo:**
- Grupo Éxito: promedio 4.20, mediana 4.0
- Grupo Fracaso: promedio 4.96, mediana 5.0
- Diferencia: -0.76 puntos en el grupo exitoso

**Tasa de éxito por nivel:**
- Carga baja (1-3): mayor tasa de éxito
- Carga media (4-6): tasa moderada
- Carga alta (7-10): **50.3% éxito** ← impacto negativo

**Insights clave:**
- Correlación cruzada: **sueño reduce carga cognitiva** (r=-0.734) — el factor más influyente
- Peor combinación: carga alta + pocas horas de código = **0% éxito**
- La carga cognitiva es un síntoma más que una causa: mejorando sueño, se mejora la carga

**Recomendación:** Controlar factores que aumentan la carga (deuda técnica, interrupciones, falta de sueño)
""",
    },
    "Plan 4": {
        "nombre": "Bugs Reportados vs Éxito",
        "correlacion": -0.178,
        "veredicto": "✅ CONFIRMADA (muy débil)",
        "color": "orange",
        "insights": "4+ bugs = 0% éxito",
        "imagenes": [
            "plan4_bugs_boxplot.png",
            "plan4_bugs_scatter.png",
            "plan4_bugs_categoria.png",
            "plan4_bugs_tasa_exito.png",
        ],
        "descripcion": [
            "Distribución por éxito",
            "Relación cantidad vs calidad",
            "Métricas comparativas",
            "Tasa de éxito por número exacto",
        ],
        "resultados": """
**Correlación de Pearson:** -0.178 (Muy débil negativa)
**Variable:** bugs_reported vs task_success

**Estadísticas por grupo:**
- Grupo Éxito: promedio 0.70 bugs, mediana 0
- Grupo Fracaso: promedio 1.10 bugs, mediana 1
- 52.2% de todas las sesiones tienen **cero bugs reportados**

**Tasa de éxito por rango:**
- Sin bugs (0): 66.7% éxito
- Pocos bugs (1-2): 57.3% éxito
- Bugs críticos (3+): reducción significativa
- 4+ bugs: **0% éxito** ← umbral de colapso

**Insights clave:**
- Es el factor con **menor correlación** de los 5 analizados individualmente
- La mayoría de sesiones exitosas tienen cero bugs, pero no es el factor determinante
- Umbral crítico en 4+ bugs: ningún desarrollador con 4+ bugs tuvo éxito en el dataset

**Recomendación:** Mantener bugs <3 por sesión; si superás 4, detener y refactorizar antes de continuar
""",
    },
    "Plan 5": {
        "nombre": "Sueño vs Éxito",
        "correlacion": 0.187,
        "veredicto": "✅ CONFIRMADA (débil)",
        "color": "orange",
        "insights": "7.1h = 92.3% éxito",
        "imagenes": [
            "plan5_sueno_boxplot.png",
            "plan5_sueno_histograma.png",
            "plan5_sueno_linea.png",
            "plan5_sueno_heatmap.png",
            "plan5_sueno_tasa_exito.png",
        ],
        "descripcion": [
            "Distribución con referencia 8h",
            "Distribución con líneas de referencia",
            "Tasa de éxito por horas exactas",
            "Interacción sueño + horas",
            "Tasa de éxito por nivel",
        ],
        "resultados": """
**Correlación de Pearson:** +0.187 (Débil positiva)
**Variable:** sleep_hours vs task_success

**Estadísticas por grupo:**
- Grupo Éxito: promedio 7.0h, mediana 7.1h
- Grupo Fracaso: promedio 5.8h, mediana 5.7h
- Diferencia: +1.2h más en el grupo exitoso

**Tasa de éxito por nivel:**
- Déficit severo (<5h): **18.6% éxito**
- Déficit moderado (5-6h): tasa baja
- Rango óptimo (7.1h): **92.3% éxito** ← pico máximo
- Exceso (>8h): ligera reducción

**Insights clave:**
- Punto óptimo exacto: **7.1 horas** de sueño maximiza el éxito
- Relación no lineal: tanto el déficit como el exceso reducen el rendimiento
- El sueño también reduce la carga cognitiva (r=-0.734 con cognitive_load)

**Recomendación:** Priorizar 7-7.5h de sueño; el déficit severo (<5h) multiplica el riesgo de fracaso
""",
    },
    "Hipótesis 6": {
        "nombre": "Uso de IA vs Éxito",
        "correlacion": 0.242,
        "veredicto": "⚠️ DÉBIL",
        "color": "blue",
        "insights": "Rango 2-4h/día = 76% éxito",
        "imagenes": [
            "plan6_uso_ia_boxplot.png",
            "plan6_uso_ia_histograma.png",
            "plan6_uso_ia_tasa_exito.png",
            "plan6_uso_ia_scatter.png",
        ],
        "descripcion": [
            "Distribución por resultado",
            "Distribución con rangos",
            "Tasa de éxito por rango",
            "Scatter IA vs horas de código",
        ],
        "resultados": """
**Correlación de Pearson:** +0.242 (Débil positiva)
**Variable:** ai_usage_hours vs task_success

**Estadísticas por grupo:**
- Grupo Éxito: promedio 1.72h/día, mediana 1.54h
- Grupo Fracaso: promedio 1.19h/día, mediana 0.95h
- Diferencia: +0.54h más en el grupo exitoso (+45.2%)
- Media global: 1.51h/día — el 72.8% de developers usa menos de 2h/día

**Tasa de éxito por rango:**
- Bajo (<2h/día / uso esporádico): **54.9% éxito** — 364 devs (72.8%)
- Medio (2-4h/día / uso moderado): **76.0% éxito** — 121 devs (24.2%) ← óptimo
- Alto (>4h/día / uso intensivo): **73.3% éxito** — 15 devs (3.0%)

**Correlaciones secundarias de ai_usage_hours:**
- vs hours_coding: +0.572 (fuerte) → ¿la IA amplifica el trabajo manual?
- vs commits: +0.370 (moderada) → mayor uso de IA asociado con más commits
- vs bugs: +0.114 (débil) → no mejora claramente la calidad del código
- vs cognitive_load: +0.120 (débil) → no reduce significativamente la carga mental

**Insights clave:**
- La correlación es positiva pero **débil**: la IA ayuda pero no es el factor principal
- El rango **óptimo es 2-4h/día**: ni muy poco ni demasiado
- Usar IA en exceso (>4h) no mejora el éxito respecto al rango medio
- La IA está fuertemente correlacionada con más horas de código (+0.572) — amplificador, no sustituto

**Recomendación:** Integrar IA de forma moderada (2-4h/día) como complemento al trabajo humano; evitar dependencia excesiva
""",
    },
    "Hipótesis 7": {
        "nombre": "Trade-off Commits vs Bugs",
        "correlacion": 0.339,
        "veredicto": "⚠️ MIXTO",
        "color": "purple",
        "insights": "Commits alto + sin bugs = 95.4% éxito",
        "imagenes": [
            "plan7_commits_bugs_scatter.png",
            "plan7_commits_bugs_tasa_exito_commits.png",
            "plan7_commits_bugs_tasa_exito_bugs.png",
            "plan7_commits_bugs_heatmap.png",
        ],
        "descripcion": [
            "Scatter commits vs bugs por éxito",
            "Tasa de éxito por rango de commits",
            "Tasa de éxito por rango de bugs",
            "Heatmap combinado commits × bugs",
        ],
        "resultados": """
**Correlaciones de Pearson:**
- commits vs task_success: **+0.339** (moderada positiva)
- bugs_reported vs task_success: **-0.178** (débil negativa)
- commits vs bugs_reported: **+0.026** (nula) ← el trade-off NO existe como correlación lineal

**Estadísticas:**
- Commits: promedio 4.61, mediana 5, rango 0-13
  - Éxito: promedio 5.35, Fracaso: promedio 3.47
- Bugs: promedio 0.86, mediana 0, rango 0-5
  - Éxito: promedio 0.70, Fracaso: promedio 1.10

**Tasa de éxito por rango de commits:**
- Bajo (<3 commits): **33.3% éxito** — n=117
- Medio (3-6 commits): **63.9% éxito** — n=266
- Alto (>6 commits): **80.3% éxito** — n=117

**Tasa de éxito por rango de bugs:**
- Sin bugs (0): **66.7% éxito** — n=261 (52.2% del dataset)
- Pocos (1-2 bugs): **57.3% éxito** — n=192
- Crítico (3+ bugs): **40.4% éxito** — n=47

**Heatmap — combinaciones clave:**

| Commits × Bugs | Tasa éxito |
|---|---|
| Alto + Sin bugs | **95.4%** ← combinación óptima |
| Alto + Pocos | 73.2% |
| Medio + Sin bugs | 65.2% |
| Medio + Crítico | 58.3% |
| Bajo + Crítico | 25.0% |
| **Alto + Crítico** | **18.2%** ← combinación peligrosa |

**Veredicto:** Trade-off **PARCIAL** — commits y bugs son estadísticamente independientes (r=+0.026), pero su combinación sí impacta fuertemente en el éxito (amplitud 77.2%)

**Recomendación:** Apuntar a commits alto + cero bugs. Si aparecen 3+ bugs, detener la cadena de commits y resolver primero
""",
    },
    "Hipótesis 8": {
        "nombre": "Balance Óptimo Multivariado",
        "correlacion": None,  # análisis multivariado, sin correlación simple
        "veredicto": "🔬 MULTIVARIADO",
        "color": "teal",
        "insights": "Cafeína alta + 6-9h código + 7h sueño = zona dorada",
        "imagenes": [
            "plan8_heatmap_cafeina_horas.png",
            "plan8_heatmap_cafeina_sueno.png",
            "plan8_heatmap_horas_sueno.png",
            "plan8_scatter_zona_dorada.png",
        ],
        "descripcion": [
            "Heatmap cafeína × horas",
            "Heatmap cafeína × sueño",
            "Heatmap horas × sueño",
            "Scatter zona dorada",
        ],
        "resultados": """
**Tipo:** Análisis Multivariado — cafeína + horas de código + sueño
**Variables combinadas:** coffee_intake_mg, hours_coding, sleep_hours
**Score compuesto ponderado por correlaciones conocidas:**
`score = 0.695 × cafeína_norm + 0.616 × horas_norm + 0.187 × sueño_norm`

**Correlaciones individuales conocidas:**
- Cafeína vs éxito: +0.695 (peso mayor)
- Horas de código vs éxito: +0.616 (peso medio)
- Sueño vs éxito: +0.187 (peso menor)

**Zona dorada identificada (combinación que maximiza éxito):**
- Cafeína: **alto (>400mg)** — rango con mayor impacto individual
- Horas de código: **medio-alto (6-9h)** — zona óptima de productividad
- Sueño: **medio (6-8h)** — sin déficit severo

**Heatmaps de pares (tasa de éxito):**
- Cafeína alta + Horas altas → máxima zona de éxito
- Cafeína baja + Horas bajas → zona de mayor fracaso
- Sueño actúa como amplificador de los otros dos factores

**Insight principal:**
La combinación de los tres factores en sus rangos óptimos genera una tasa de éxito superior a la de cualquier factor analizado individualmente.
Un score compuesto ponderado tiene mayor poder predictivo que cada variable por separado.

**Recomendación práctica (zona dorada):**
→ Dormir ~7h + consumir >400mg de cafeína + codificar 6-9h = combinación de máximo rendimiento según este dataset
""",
    },
}

# ─── Sidebar para navegación ───────────────────────────────────────────────────
st.sidebar.title("🧭 Navegación")

seccion = st.sidebar.selectbox(
    "Selecciona una sección:",
    [
        "📊 Resumen General",
        "🎯 Análisis por Hipótesis",
        "📈 Análisis Comparativo",
        "🔍 Dataset Explorer",
        "📋 Información del Proyecto",
        "🤖 Predictor de Éxito",
        "📊 Modelo Predictivo",
    ],
)

# ─── Helper: carga del pipeline ML ───────────────────────────────────────────


@st.cache_resource
def cargar_pipeline():
    """Carga el pipeline sklearn (scaler + RandomForest) desde disco."""
    try:
        pipe = joblib.load("dashboard/model/pipeline.pkl")
        features = joblib.load("dashboard/model/feature_names.pkl")
        with open("dashboard/model/model_metadata.json") as f:
            meta = json_lib.load(f)
        return pipe, features, meta
    except FileNotFoundError:
        return None, None, None


def calcular_features_derivadas(
    hours_coding, coffee_intake_mg, sleep_hours, cognitive_load, commits
):
    """Calcula las 6 features derivadas con la misma lógica de fase3."""
    sleep_deficit = 8.0 - sleep_hours
    productivity_ratio = commits / max(hours_coding, 0.1)
    caffeine_per_hour = coffee_intake_mg / max(hours_coding, 0.1)
    work_intensity = hours_coding * cognitive_load
    coffee_category = (
        0 if coffee_intake_mg < 200 else (1 if coffee_intake_mg <= 400 else 2)
    )
    sleep_category = 0 if sleep_hours < 6 else (1 if sleep_hours <= 8 else 2)
    return {
        "sleep_deficit": sleep_deficit,
        "productivity_ratio": productivity_ratio,
        "caffeine_per_hour": caffeine_per_hour,
        "work_intensity": work_intensity,
        "coffee_category": coffee_category,
        "sleep_category": sleep_category,
    }


# Etiquetas de features en español — usado en secciones ML
LABELS = {
    "hours_coding": "Horas de Código",
    "coffee_intake_mg": "Cafeína (mg)",
    "distractions": "Distracciones",
    "sleep_hours": "Horas de Sueño",
    "commits": "Commits",
    "bugs_reported": "Bugs Reportados",
    "ai_usage_hours": "Uso de IA (h)",
    "cognitive_load": "Carga Cognitiva",
    "sleep_deficit": "Déficit de Sueño",
    "productivity_ratio": "Ratio Productividad",
    "caffeine_per_hour": "Cafeína por Hora",
    "work_intensity": "Intensidad de Trabajo",
    "coffee_category": "Categoría Cafeína",
    "sleep_category": "Categoría Sueño",
}

# ─── Resumen General ───────────────────────────────────────────────────────────
if seccion == "📊 Resumen General":
    st.header("📊 Resumen General de Hipótesis")

    # Tabla de resumen — usa display_label para mostrar "Hipótesis X" en vez de "Plan X"
    resumen_data = []
    for plan, data in hipotesis_data.items():
        corr_str = (
            f"{data['correlacion']:.3f}" if data["correlacion"] is not None else "N/A"
        )
        impacto_str = (
            f"{data['correlacion']:.1%}" if data["correlacion"] is not None else "N/A"
        )
        resumen_data.append(
            {
                "Hipótesis": display_label(plan),
                "Nombre": data["nombre"],
                "Correlación": corr_str,
                "Veredicto": data["veredicto"],
                "Impacto": impacto_str,
                "Insights Clave": data["insights"],
            }
        )

    df_resumen = pd.DataFrame(resumen_data)

    # Ordenar por correlación descendente (N/A queda al final)
    def _sort_corr(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return float("-inf")

    df_resumen = df_resumen.sort_values(
        "Correlación", ascending=False, key=lambda col: col.map(_sort_corr)
    )

    st.dataframe(df_resumen, use_container_width=True, hide_index=True)

    # Gráfico de correlaciones — saltea hipótesis sin correlación simple
    st.subheader("📈 Ranking de Impacto en Task Success")

    fig_corr = go.Figure()

    planes_ordenados = sorted(
        hipotesis_data.items(),
        key=lambda x: (
            x[1]["correlacion"] if x[1]["correlacion"] is not None else float("-inf")
        ),
        reverse=True,
    )

    for plan, data in planes_ordenados:
        # Saltear análisis multivariado (sin correlación simple)
        if data["correlacion"] is None:
            continue
        fig_corr.add_trace(
            go.Bar(
                name=data["nombre"],
                x=[data["correlacion"]],
                y=[data["nombre"]],
                orientation="h",
                marker_color=data["color"] if data["color"] == "green" else "orange",
                text=[f"{data['correlacion']:.3f}"],
                textposition="auto",
            )
        )

    fig_corr.update_layout(
        title="Correlación con Task Success",
        xaxis_title="Correlación de Pearson",
        yaxis_title="Hipótesis",
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Métricas clave — valores dinámicos
    st.subheader("🎯 Métricas Clave del Proyecto")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Hipótesis", len(hipotesis_data))
    with col2:
        confirmadas = sum(
            1 for data in hipotesis_data.values() if "CONFIRMADA" in data["veredicto"]
        )
        st.metric("Hipótesis Confirmadas", f"{confirmadas}/{len(hipotesis_data)}")
    with col3:
        max_corr = max(
            (
                data["correlacion"]
                for data in hipotesis_data.values()
                if data["correlacion"] is not None
            ),
            default=0,
        )
        st.metric("Correlación Máxima", f"{max_corr:.3f}")
    with col4:
        total_imagenes = sum(len(data["imagenes"]) for data in hipotesis_data.values())
        st.metric("Total Visualizaciones", f"{total_imagenes}")

# ─── Análisis por Hipótesis ────────────────────────────────────────────────────
elif seccion == "🎯 Análisis por Hipótesis":
    st.header("🎯 Análisis Detallado por Hipótesis")

    # Selector de hipótesis — muestra "Hipótesis X" en el dropdown
    plan_seleccionado = st.selectbox(
        "Selecciona una hipótesis para analizar:",
        list(hipotesis_data.keys()),
        format_func=lambda x: f"{display_label(x)} - {hipotesis_data[x]['nombre']}",
    )

    data = hipotesis_data[plan_seleccionado]

    # Información de la hipótesis
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader(f"📊 {data['nombre']}")
        corr_display = (
            f"{data['correlacion']:.3f}"
            if data["correlacion"] is not None
            else "N/A (análisis multivariado)"
        )
        st.markdown(f"**Correlación:** `{corr_display}`")
        st.markdown(f"**Veredicto:** {data['veredicto']}")
        st.markdown(f"**Insights Clave:** {data['insights']}")

    with col2:
        corr_metric = (
            f"{data['correlacion']:.3f}" if data["correlacion"] is not None else "N/A"
        )
        st.metric("Correlación", corr_metric)

    with col3:
        st.metric("Visualizaciones", len(data["imagenes"]))

    st.markdown("---")

    # Resultados detallados del análisis
    st.subheader("📋 Resultados del Análisis")
    st.markdown(data["resultados"])
    st.markdown("---")

    # Galería de imágenes
    st.subheader(f"🖼️ Galería de Visualizaciones - {display_label(plan_seleccionado)}")

    imagen_seleccionada = st.selectbox(
        "Selecciona una visualización:",
        list(zip(data["descripcion"], data["imagenes"])),
        format_func=lambda x: x[0],
    )

    ruta_imagen = os.path.join("dashboard/assets", imagen_seleccionada[1])
    if os.path.exists(ruta_imagen):
        image = Image.open(ruta_imagen)
        st.image(image, caption=imagen_seleccionada[0], use_container_width=True)
    else:
        st.error(f"No se encontró la imagen: {ruta_imagen}")

    # Miniaturas de todas las imágenes
    st.subheader("📸 Todas las Visualizaciones")

    cols = st.columns(len(data["imagenes"]))
    for i, (desc, img) in enumerate(zip(data["descripcion"], data["imagenes"])):
        with cols[i]:
            ruta_mini = os.path.join("dashboard/assets", img)
            if os.path.exists(ruta_mini):
                image = Image.open(ruta_mini)
                st.image(image, caption=desc, use_container_width=True)
            else:
                st.error(f"No encontrada: {img}")

# ─── Análisis Comparativo ──────────────────────────────────────────────────────
elif seccion == "📈 Análisis Comparativo":
    st.header("📈 Análisis Comparativo de Todas las Variables")

    # Tabla comparativa completa
    st.subheader("🔍 Comparación Detallada")

    comparativa_data = []
    for plan, data in hipotesis_data.items():
        corr = data["correlacion"]
        corr_abs = abs(corr) if corr is not None else None

        if corr is None:
            tipo = "Multivariado"
            fuerza = "N/A"
        elif corr > 0:
            tipo = "Positiva"
            fuerza = (
                "Fuerte"
                if corr_abs > 0.5
                else "Moderada"
                if corr_abs > 0.3
                else "Débil"
                if corr_abs > 0.1
                else "Muy Débil"
            )
        else:
            tipo = "Negativa"
            fuerza = (
                "Fuerte"
                if corr_abs > 0.5
                else "Moderada"
                if corr_abs > 0.3
                else "Débil"
                if corr_abs > 0.1
                else "Muy Débil"
            )

        comparativa_data.append(
            {
                "Hipótesis": display_label(plan),
                "Nombre": data["nombre"],
                "Correlación": corr if corr is not None else "N/A",
                "Veredicto": data["veredicto"],
                "Tipo": tipo,
                "Fuerza": fuerza,
                "Visualizaciones": len(data["imagenes"]),
            }
        )

    df_comparativa = pd.DataFrame(comparativa_data)
    st.dataframe(df_comparativa, use_container_width=True, hide_index=True)

    # Gráfico de barras comparativo — saltea hipótesis sin correlación simple
    st.subheader("📊 Comparación de Correlaciones")

    fig_comp = go.Figure()

    for plan, data in hipotesis_data.items():
        if data["correlacion"] is None:
            continue
        color = "green" if data["correlacion"] > 0 else "red"
        fig_comp.add_trace(
            go.Bar(
                name=data["nombre"],
                x=[display_label(plan)],
                y=[data["correlacion"]],
                marker_color=color,
                text=[f"{data['correlacion']:.3f}"],
                textposition="auto",
            )
        )

    fig_comp.update_layout(
        title="Correlaciones de Todas las Hipótesis",
        xaxis_title="Hipótesis",
        yaxis_title="Correlación con Task Success",
        height=500,
    )

    # Línea base en y=0
    fig_comp.add_hline(y=0, line_dash="dash", line_color="black")

    st.plotly_chart(fig_comp, use_container_width=True)

    # Análisis de fuerza — saltea análisis multivariado
    st.subheader("💪 Análisis de Fuerza de Correlaciones")

    fuerza_data = []
    for plan, data in hipotesis_data.items():
        if data["correlacion"] is None:
            continue
        fuerza_abs = abs(data["correlacion"])
        if fuerza_abs > 0.5:
            categoria = "Fuerte"
        elif fuerza_abs > 0.3:
            categoria = "Moderada"
        elif fuerza_abs > 0.1:
            categoria = "Débil"
        else:
            categoria = "Muy Débil"

        fuerza_data.append(
            {
                "Hipótesis": display_label(plan),
                "Nombre": data["nombre"],
                "Fuerza Absoluta": fuerza_abs,
                "Categoría": categoria,
                "Dirección": "Positiva" if data["correlacion"] > 0 else "Negativa",
            }
        )

    df_fuerza = pd.DataFrame(fuerza_data)

    fig_fuerza = px.bar(
        df_fuerza,
        x="Hipótesis",
        y="Fuerza Absoluta",
        color="Dirección",
        title="Fuerza de Correlación por Hipótesis",
        text="Fuerza Absoluta",
        hover_data=["Nombre", "Categoría"],
    )

    fig_fuerza.update_layout(height=500)
    st.plotly_chart(fig_fuerza, use_container_width=True)

# ─── Dataset Explorer ──────────────────────────────────────────────────────────
elif seccion == "🔍 Dataset Explorer":
    st.header("🔍 Dataset Explorer")

    try:
        df = pd.read_csv("dashboard/data/ai_dev_productivity.csv")
        st.success("✅ Dataset cargado exitosamente")

        # Estadísticas básicas
        st.subheader("📊 Estadísticas del Dataset")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de Registros", len(df))
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            success_rate = df["task_success"].mean() * 100
            st.metric("Tasa de Éxito", f"{success_rate:.1f}%")
        with col4:
            st.metric("Variables Analizadas", "8")

        # Vista previa del dataset
        st.subheader("👁️ Vista Previa del Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # Filtros interactivos
        st.subheader("🔍 Filtros Interactivos")

        col1, col2 = st.columns(2)

        with col1:
            exito_filtro = st.selectbox(
                "Filtrar por Task Success:",
                ["Todos", "Éxito (1)", "Fracaso (0)"],
            )

            if exito_filtro == "Éxito (1)":
                df_filtrado = df[df["task_success"] == 1]
            elif exito_filtro == "Fracaso (0)":
                df_filtrado = df[df["task_success"] == 0]
            else:
                df_filtrado = df

        with col2:
            horas_min, horas_max = st.slider(
                "Rango de Horas de Código:",
                float(df["hours_coding"].min()),
                float(df["hours_coding"].max()),
                (float(df["hours_coding"].min()), float(df["hours_coding"].max())),
            )

            df_filtrado = df_filtrado[
                (df_filtrado["hours_coding"] >= horas_min)
                & (df_filtrado["hours_coding"] <= horas_max)
            ]

        # Estadísticas del dataset filtrado
        st.subheader(
            f"📈 Estadísticas del Dataset Filtrado ({len(df_filtrado)} registros)"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Registros Filtrados", len(df_filtrado))
        with col2:
            if len(df_filtrado) > 0:
                success_filtrado = df_filtrado["task_success"].mean() * 100
                st.metric("Tasa de Éxito Filtrada", f"{success_filtrado:.1f}%")
            else:
                st.metric("Tasa de Éxito Filtrada", "N/A")
        with col3:
            if len(df_filtrado) > 0:
                st.metric("Promedio Horas", f"{df_filtrado['hours_coding'].mean():.1f}")
            else:
                st.metric("Promedio Horas", "N/A")

        # Dataset filtrado
        if len(df_filtrado) > 0:
            st.dataframe(df_filtrado, use_container_width=True)
        else:
            st.warning("No hay registros que coincidan con los filtros seleccionados")

        # Distribuciones — incluye variables de hipótesis 6 y 7
        st.subheader("📊 Distribuciones de Variables Clave")

        variable_dist = st.selectbox(
            "Selecciona variable para visualizar:",
            [
                "hours_coding",
                "coffee_intake_mg",
                "sleep_hours",
                "cognitive_load",
                "bugs_reported",
                "ai_usage_hours",
                "commits",
            ],
        )

        fig_dist = px.histogram(
            df_filtrado if len(df_filtrado) > 0 else df,
            x=variable_dist,
            color="task_success",
            title=f"Distribución de {variable_dist}",
            barmode="overlay",
        )

        st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error al cargar el dataset: {e}")

# ─── Información del Proyecto ──────────────────────────────────────────────────
elif seccion == "📋 Información del Proyecto":
    st.header("📋 Información del Proyecto")

    # Información general
    st.subheader("🎯 Objetivo del Proyecto")
    st.markdown(
        """
    Analizar el impacto de diferentes factores en la productividad y éxito de los desarrolladores
    utilizando técnicas de Big Data con PySpark y visualizaciones interactivas.
    """
    )

    # Metodología — visualizaciones calculadas dinámicamente
    total_viz = sum(len(data["imagenes"]) for data in hipotesis_data.values())

    st.subheader("🔬 Metodología")
    st.markdown(
        f"""
    - **Tecnología:** PySpark + Pandas + Streamlit
    - **Dataset:** 500 registros de productividad de desarrolladores
    - **Análisis:** Correlación de Pearson + análisis por rangos + análisis multivariado
    - **Visualizaciones:** {total_viz} gráficos generados con Matplotlib/Seaborn
    """
    )

    # Stack tecnológico
    st.subheader("🛠️ Stack Tecnológico")

    tech_data = {
        "Tecnología": [
            "Python",
            "PySpark",
            "Pandas",
            "Streamlit",
            "Matplotlib",
            "Seaborn",
        ],
        "Versión": ["3.14", "3.5.0", "2.3.3", "1.55.0", "3.10.8", "0.13.2"],
        "Propósito": [
            "Lenguaje principal",
            "Procesamiento Big Data",
            "Manipulación datos",
            "Dashboard interactivo",
            "Visualización",
            "Visualización estadística",
        ],
    }

    df_tech = pd.DataFrame(tech_data)
    st.dataframe(df_tech, use_container_width=True, hide_index=True)

    # Estructura del proyecto
    st.subheader("📁 Estructura del Proyecto")

    st.markdown(
        f"""
    ```
    BigDataProject/
    ├── data/
    │   └── ai_dev_productivity.csv
    ├── notebooks/
    │   ├── planX_*_analysis.py (8 scripts)
    │   └── results/
    │       ├── plan1-cafeina/
    │       ├── plan2-horas-codigo/
    │       ├── plan3-carga-cognitiva/
    │       ├── plan4-bugs-reportados/
    │       ├── plan5-sueno/
    │       ├── plan6-uso-ia/
    │       ├── plan7-commits-bugs/
    │       └── plan8-balance-multivariado/
    ├── dashboard/
    │   ├── dashboard.py
    │   ├── assets/ ({total_viz} imágenes)
    │   └── data/
    ├── docs/
    │   ├── SPECS.md
    │   ├── README.md
    │   ├── CHANGELOG.md
    │   └── ARCHITECTURE.md
    └── .windsurf/
        └── plans/ (8 planes de análisis)
    ```
    """
    )

    # Estándar de calidad
    st.subheader("⭐ Estándar de Calidad")

    st.markdown(
        """
    **Requisitos cumplidos para todos los planes:**
    - ✅ 8 secciones en archivos de estadísticas
    - ✅ 4 requisitos mínimos para gráficos
    - ✅ Formato estandarizado de archivos
    - ✅ Documentación completa y comprensible
    """
    )

    # Conclusiones
    st.subheader("🎉 Conclusiones del Proyecto")

    st.markdown(
        f"""
    **Hallazgos Principales:**
    1. **Cafeína** es el factor más influyente (r=0.695)
    2. **Horas de código** tienen fuerte correlación positiva (r=0.616)
    3. **Carga cognitiva** impacta negativamente pero débilmente (r=-0.200)
    4. **Bugs reportados** tienen impacto mínimo (r=-0.178)
    5. **Sueño** muestra correlación positiva débil (r=0.187)
    6. **Uso de IA** es un amplificador moderado — óptimo 2-4h/día (r=0.242)
    7. **Trade-off Commits vs Bugs** — independientes entre sí pero su combinación impacta fuertemente (r=0.339)
    8. **Balance Multivariado** — zona dorada: cafeína alta + 6-9h código + 7h sueño

    **Impacto del Proyecto:**
    - {len(hipotesis_data)} hipótesis analizadas y validadas
    - {total_viz} visualizaciones generadas
    - 1 dashboard interactivo para exploración
    - Evidencia cuantitativa para optimizar productividad
    """
    )

# ─── Modelo Predictivo ────────────────────────────────────────────────────────
elif seccion == "📊 Modelo Predictivo":
    st.header("📊 Modelo Predictivo — Random Forest")

    # Cargar metadata
    try:
        with open("dashboard/model/model_metadata.json") as f:
            meta = json_lib.load(f)
    except FileNotFoundError:
        st.error(
            "❌ Metadata del modelo no encontrada. Ejecutá primero `fase45_modelado_evaluacion.py`."
        )
        st.stop()

    # Métricas en cards
    st.subheader("🎯 Métricas en Test Set (datos nunca vistos durante entrenamiento)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{meta['test_accuracy'] * 100:.1f}%")
    col2.metric("Precision", f"{meta['test_precision'] * 100:.1f}%")
    col3.metric("Recall", f"{meta['test_recall'] * 100:.1f}%")
    col4.metric("F1-Score", f"{meta['test_f1'] * 100:.1f}%")
    col5.metric("ROC-AUC", f"{meta['test_roc_auc']:.3f}")

    # Advertencia honesta sobre el dataset
    st.info("""
    ⚠️ **Nota académica:** El accuracy de 98.7% es muy alto porque el dataset es sintético
    y sus patrones son muy marcados. En un dataset de producción real los valores serían menores.
    El modelo sigue siendo válido para demostrar las técnicas de ML aplicadas.
    """)

    # 4 imágenes en 2×2
    st.subheader("📈 Visualizaciones del Modelo")

    imagenes_modelo = [
        ("fase45_model_comparison.png", "Comparación de Modelos (Val Set)"),
        ("fase45_confusion_matrix.png", "Matriz de Confusión (Test Set)"),
        ("fase45_roc_curve.png", "Curva ROC (Test Set)"),
        ("fase45_feature_importance.png", "Importancia de Features"),
    ]

    col1, col2 = st.columns(2)
    for i, (img, titulo) in enumerate(imagenes_modelo):
        with col1 if i % 2 == 0 else col2:
            ruta = os.path.join("dashboard/assets", img)
            if os.path.exists(ruta):
                st.image(Image.open(ruta), caption=titulo, use_container_width=True)

    # Feature importance como tabla interactiva
    st.subheader("🏆 Ranking de Importancia de Features")

    fi = meta["feature_importance"]
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
    df_fi = pd.DataFrame(
        [
            {
                "Feature": LABELS.get(k, k),
                "Importancia": v,
                "Porcentaje": f"{v * 100:.1f}%",
            }
            for k, v in fi_sorted
        ]
    )
    st.dataframe(df_fi, use_container_width=True, hide_index=True)

    st.subheader("📋 Descripción del Proceso")
    st.markdown("""
    **Algoritmos evaluados:**
    - **Logistic Regression** (baseline) — 89.3% accuracy en validación
    - **Decision Tree** — 100% en validación (estructura interpretable)
    - **Random Forest** ← modelo seleccionado — 100% en validación, 98.7% en test

    **Optimización:** GridSearchCV con 5-fold cross-validation sobre F1-score

    **Split del dataset:** 70% train / 15% validación / 15% test (estratificado)

    **SMOTE:** Aplicado al training set para balancear las clases (la clase minoritaria era <40%)

    **Features totales:** 14 (8 originales + 6 derivadas por feature engineering)
    """)

# ─── Predictor de Éxito ───────────────────────────────────────────────────────
elif seccion == "🤖 Predictor de Éxito":
    st.header("🤖 Predictor de Éxito en Tareas de Desarrollo")

    pipeline, feature_names, meta = cargar_pipeline()

    if pipeline is None:
        st.error(
            "❌ Modelo no encontrado. Ejecutá primero `fase45_modelado_evaluacion.py`."
        )
        st.stop()

    # ── Bloque 1: Cómo funciona ──
    with st.expander("📖 ¿Cómo funciona este predictor?", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Algoritmo", "Random Forest")
        col2.metric("Accuracy", f"{meta['test_accuracy'] * 100:.1f}%")
        col3.metric("Features usadas", "14")

        st.markdown("""
        Este predictor usa un modelo **Random Forest** entrenado sobre 500 registros de productividad
        de desarrolladores. Tomás tus valores actuales, el modelo analiza 14 factores y predice si
        tu sesión de trabajo terminará en éxito o fracaso.

        **¿Qué hace internamente?**
        1. Recibe tus 8 valores originales (los que ingresás abajo)
        2. Calcula 6 variables derivadas automáticamente (déficit de sueño, intensidad de trabajo, etc.)
        3. Normaliza los 14 valores con el mismo scaler del entrenamiento
        4. El Random Forest vota entre 100 árboles de decisión
        5. Devuelve la predicción + probabilidad de éxito

        **Factores más importantes** (según el modelo entrenado):
        """)

        fi = meta["feature_importance"]
        top3 = sorted(fi.items(), key=lambda x: -x[1])[:3]
        for k, v in top3:
            st.progress(v, text=f"**{LABELS.get(k, k)}** — {v * 100:.1f}%")

        st.warning("""
        ⚠️ **Limitación importante:** Este modelo fue entrenado en un dataset sintético.
        Las predicciones son una aproximación educativa, no una garantía de rendimiento real.
        Correlación no implica causalidad.
        """)

    st.markdown("---")

    # ── Bloque 2: Inputs del usuario ──
    st.subheader("📝 Ingresá tus valores de hoy")
    st.markdown(
        "Mové los sliders según tu situación actual. La predicción se actualiza en tiempo real."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⌨️ Trabajo**")
        hours_coding = st.slider(
            "Horas de código hoy",
            0.0,
            16.0,
            6.0,
            0.5,
            help="¿Cuántas horas programaste o pensás programar hoy?",
        )
        commits = st.slider(
            "Commits realizados",
            0,
            20,
            5,
            1,
            help="Número de commits en la sesión de hoy",
        )
        bugs_reported = st.slider(
            "Bugs encontrados",
            0,
            10,
            1,
            1,
            help="Bugs detectados o reportados durante la sesión",
        )
        distractions = st.slider(
            "Número de distracciones",
            0,
            15,
            3,
            1,
            help="Interrupciones durante el trabajo (reuniones, mensajes, etc.)",
        )

    with col2:
        st.markdown("**🧠 Estado personal**")
        sleep_hours = st.slider(
            "Horas de sueño anoche",
            3.0,
            12.0,
            7.0,
            0.5,
            help="¿Cuántas horas dormiste la noche anterior?",
        )
        coffee_intake_mg = st.slider(
            "Cafeína consumida (mg)",
            0,
            800,
            300,
            25,
            help="1 taza de café ≈ 95mg | 1 espresso ≈ 63mg | 1 Red Bull ≈ 80mg",
        )
        cognitive_load = st.slider(
            "Carga cognitiva percibida (1-10)",
            1,
            10,
            5,
            1,
            help="¿Qué tan compleja o exigente te parece la tarea de hoy?",
        )
        ai_usage_hours = st.slider(
            "Horas usando herramientas de IA",
            0.0,
            8.0,
            1.0,
            0.5,
            help="GitHub Copilot, ChatGPT, Claude, Cursor, etc.",
        )

    # ── Bloque 3: Features derivadas (educativo) ──
    derivadas = calcular_features_derivadas(
        hours_coding, coffee_intake_mg, sleep_hours, cognitive_load, commits
    )

    with st.expander("🔬 Variables calculadas automáticamente por el modelo"):
        st.markdown(
            "El modelo no usa solo tus valores directos — también calcula estas variables derivadas:"
        )
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Déficit de sueño",
            f"{derivadas['sleep_deficit']:.1f}h",
            help="8h recomendadas − tus horas de sueño",
        )
        d2.metric(
            "Ratio de productividad",
            f"{derivadas['productivity_ratio']:.2f}",
            help="Commits por hora de código",
        )
        d3.metric(
            "Cafeína por hora",
            f"{derivadas['caffeine_per_hour']:.1f} mg/h",
            help="Cafeína consumida ÷ horas de código",
        )

        d4, d5, d6 = st.columns(3)
        d4.metric(
            "Intensidad de trabajo",
            f"{derivadas['work_intensity']:.1f}",
            help="Horas de código × carga cognitiva",
        )
        cat_cafe = ["Bajo (<200mg)", "Medio (200-400mg)", "Alto (>400mg)"][
            derivadas["coffee_category"]
        ]
        cat_sueno = ["Insuficiente (<6h)", "Óptimo (6-8h)", "Excesivo (>8h)"][
            derivadas["sleep_category"]
        ]
        d5.metric("Categoría cafeína", cat_cafe)
        d6.metric("Categoría sueño", cat_sueno)

    st.markdown("---")

    # ── Bloque 4: Predicción ──
    st.subheader("🎯 Predicción")

    # Construir vector de 14 features en el orden exacto
    input_vector = pd.DataFrame(
        [
            [
                hours_coding,
                coffee_intake_mg,
                distractions,
                sleep_hours,
                commits,
                bugs_reported,
                ai_usage_hours,
                cognitive_load,
                derivadas["sleep_deficit"],
                derivadas["productivity_ratio"],
                derivadas["caffeine_per_hour"],
                derivadas["work_intensity"],
                derivadas["coffee_category"],
                derivadas["sleep_category"],
            ]
        ],
        columns=feature_names,
    )

    # Predicción
    pred = pipeline.predict(input_vector)[0]
    prob = pipeline.predict_proba(input_vector)[0]
    prob_exito = prob[1]
    prob_fracaso = prob[0]

    # Resultado visual
    col_pred, col_prob = st.columns([1, 2])

    with col_pred:
        if pred == 1:
            st.success("## ✅ ÉXITO")
            st.markdown("El modelo predice que **tu sesión será exitosa**.")
        else:
            st.error("## ❌ FRACASO")
            st.markdown("El modelo predice que **tu sesión NO será exitosa**.")

    with col_prob:
        st.markdown("**Probabilidad de éxito:**")
        st.progress(float(prob_exito))
        st.markdown(f"### {prob_exito * 100:.1f}% de probabilidad de éxito")
        if prob_exito >= 0.8:
            st.markdown(
                "🟢 **Alta confianza** — el modelo está muy seguro de su predicción"
            )
        elif prob_exito >= 0.5:
            st.markdown(
                "🟡 **Confianza moderada** — la sesión podría ir en cualquier dirección"
            )
        else:
            st.markdown("🔴 **Baja confianza en éxito** — múltiples factores en contra")

    # Top 5 factores más influyentes para ESTA predicción
    st.markdown("---")
    st.subheader("💡 ¿Qué factores pesaron más en esta predicción?")
    st.markdown("*Importancia global del modelo × magnitud de tu valor normalizado:*")

    input_vals = input_vector.iloc[0].to_dict()
    fi = meta["feature_importance"]
    scores = {k: fi.get(k, 0) * abs(input_vals.get(k, 0)) for k in feature_names}
    top_factors = sorted(scores.items(), key=lambda x: -x[1])[:5]

    for k, score in top_factors:
        label = LABELS.get(k, k)
        val = input_vals.get(k, 0)
        importancia = fi.get(k, 0)
        st.markdown(
            f"**{label}** — valor: `{val:.2f}` | importancia global: `{importancia * 100:.1f}%`"
        )
        st.progress(min(float(importancia), 1.0))

    # Recomendaciones personalizadas
    st.markdown("---")
    st.subheader("📌 Recomendaciones basadas en tus valores")
    recomendaciones = []

    if hours_coding < 3:
        recomendaciones.append(
            "⚠️ **Horas de código bajas** (<3h): el modelo indica que menos de 3h tiene correlación con fracaso. Considerá extender la sesión."
        )
    if coffee_intake_mg < 200:
        recomendaciones.append(
            "☕ **Cafeína baja** (<200mg): el análisis previo mostró 0% de éxito en este rango. Considerá aumentar el consumo."
        )
    if sleep_hours < 6:
        recomendaciones.append(
            "😴 **Déficit de sueño severo** (<6h): el sueño insuficiente eleva la carga cognitiva y reduce el éxito en 40%+."
        )
    if cognitive_load >= 8:
        recomendaciones.append(
            "🧠 **Carga cognitiva muy alta** (≥8): considerá dividir la tarea en partes más pequeñas."
        )
    if bugs_reported >= 4:
        recomendaciones.append(
            "🐛 **Bugs críticos** (≥4): el análisis mostró 0% de éxito con 4+ bugs. Detené y refactorizá antes de continuar."
        )
    if distractions >= 8:
        recomendaciones.append(
            "📵 **Muchas distracciones** (≥8): reducir interrupciones tiene impacto directo en la productividad."
        )

    if recomendaciones:
        for r in recomendaciones:
            st.warning(r)
    else:
        st.success(
            "✅ Tus valores están dentro de los rangos asociados con mayor éxito. ¡Seguí así!"
        )

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "🚀 **BigData Project Dashboard** | Desarrollado con Streamlit | PySpark + Pandas + Python"
)

# Información de ejecución en el sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.markdown(
        f"**Total de imágenes:** {sum(len(data['imagenes']) for data in hipotesis_data.values())}"
    )
    st.markdown(f"**Hipótesis analizadas:** {len(hipotesis_data)}")
    st.markdown("**Status:** ✅ Todos los análisis completados")

    if st.button("🔄 Recargar Dashboard"):
        st.rerun()
