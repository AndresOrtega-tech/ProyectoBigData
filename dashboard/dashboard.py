import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="BigData Project - Dashboard de Productividad",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚀 BigData Project - Dashboard de Productividad de Desarrolladores")
st.markdown("### Análisis de 5 hipótesis con PySpark + Streamlit")
st.markdown("---")

# Datos de las hipótesis
hipotesis_data = {
    "Plan 1": {
        "nombre": "Cafeína vs Éxito",
        "correlacion": 0.695,
        "veredicto": "✅ CONFIRMADA",
        "color": "green",
        "insights": ">400mg = 83.6% éxito",
        "imagenes": ["plan1_cafeina_boxplot.png", "plan1_cafeina_histograma.png", "plan1_cafeina_tasa_exito.png"],
        "descripcion": ["Distribución con anotaciones", "Patrones con rangos visibles", "Relación dose-respuesta"]
    },
    "Plan 2": {
        "nombre": "Horas de Código vs Éxito",
        "correlacion": 0.616,
        "veredicto": "✅ CONFIRMADA",
        "color": "green",
        "insights": "6-9h = 85.7% éxito",
        "imagenes": ["plan2_horas_boxplot.png", "plan2_horas_histograma.png", "plan2_horas_tasa_exito.png"],
        "descripcion": ["Distribución con medianas", "Patrones con líneas de referencia", "Tasa de éxito por rango"]
    },
    "Plan 3": {
        "nombre": "Carga Cognitiva vs Éxito",
        "correlacion": -0.200,
        "veredicto": "✅ CONFIRMADA (débil)",
        "color": "orange",
        "insights": "Alta carga = 50.3% éxito",
        "imagenes": ["plan3_cognitiva_boxplot.png", "plan3_cognitiva_scatter.png", "plan3_cognitiva_heatmap.png", "plan3_cognitiva_tasa_exito.png"],
        "descripcion": ["Distribución con medianas", "Interacción carga vs horas", "Tasa de éxito por combinación", "Tasa de éxito por nivel"]
    },
    "Plan 4": {
        "nombre": "Bugs Reportados vs Éxito",
        "correlacion": -0.178,
        "veredicto": "✅ CONFIRMADA (muy débil)",
        "color": "orange",
        "insights": "4+ bugs = 0% éxito",
        "imagenes": ["plan4_bugs_boxplot.png", "plan4_bugs_scatter.png", "plan4_bugs_categoria.png", "plan4_bugs_tasa_exito.png"],
        "descripcion": ["Distribución por éxito", "Relación cantidad vs calidad", "Métricas comparativas", "Tasa de éxito por número exacto"]
    },
    "Plan 5": {
        "nombre": "Sueño vs Éxito",
        "correlacion": 0.187,
        "veredicto": "✅ CONFIRMADA (débil)",
        "color": "orange",
        "insights": "7.1h = 92.3% éxito",
        "imagenes": ["plan5_sueno_boxplot.png", "plan5_sueno_histograma.png", "plan5_sueno_linea.png", "plan5_sueno_heatmap.png", "plan5_sueno_tasa_exito.png"],
        "descripcion": ["Distribución con referencia 8h", "Distribución con líneas de referencia", "Tasa de éxito por horas exactas", "Interacción sueño + horas", "Tasa de éxito por nivel"]
    }
}

# Sidebar para navegación
st.sidebar.title("🧭 Navegación")

# Sección seleccionada
seccion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["📊 Resumen General", "🎯 Análisis por Plan", "📈 Análisis Comparativo", "🔍 Dataset Explorer", "📋 Información del Proyecto"]
)

if seccion == "📊 Resumen General":
    st.header("📊 Resumen General de Hipótesis")
    
    # Crear tabla de resumen
    resumen_data = []
    for plan, data in hipotesis_data.items():
        resumen_data.append({
            "Plan": plan,
            "Hipótesis": data["nombre"],
            "Correlación": f"{data['correlacion']:.3f}",
            "Veredicto": data["veredicto"],
            "Impacto": f"{data['correlacion']:.1%}",
            "Insights Clave": data["insights"]
        })
    
    df_resumen = pd.DataFrame(resumen_data)
    
    # Ordenar por correlación (descendente)
    df_resumen = df_resumen.sort_values("Correlación", ascending=False, key=lambda x: pd.to_numeric(x.str.replace("%", "")))
    
    # Mostrar tabla
    st.dataframe(df_resumen, use_container_width=True, hide_index=True)
    
    # Gráfico de correlaciones
    st.subheader("📈 Ranking de Impacto en Task Success")
    
    fig_corr = go.Figure()
    
    # Ordenar planes por correlación
    planes_ordenados = sorted(hipotesis_data.items(), key=lambda x: x[1]["correlacion"], reverse=True)
    
    for plan, data in planes_ordenados:
        fig_corr.add_trace(go.Bar(
            name=data["nombre"],
            x=[data["correlacion"]],
            y=[data["nombre"]],
            orientation='h',
            marker_color=data["color"] if data["color"] == "green" else "orange",
            text=[f"{data['correlacion']:.3f}"],
            textposition='auto',
        ))
    
    fig_corr.update_layout(
        title="Correlación con Task Success",
        xaxis_title="Correlación de Pearson",
        yaxis_title="Hipótesis",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Métricas clave
    st.subheader("🎯 Métricas Clave del Proyecto")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Planes", "5")
    with col2:
        confirmadas = sum(1 for data in hipotesis_data.values() if "CONFIRMADA" in data["veredicto"])
        st.metric("Hipótesis Confirmadas", f"{confirmadas}/5")
    with col3:
        max_corr = max(data["correlacion"] for data in hipotesis_data.values())
        st.metric("Correlación Máxima", f"{max_corr:.3f}")
    with col4:
        total_imagenes = sum(len(data["imagenes"]) for data in hipotesis_data.values())
        st.metric("Total Visualizaciones", f"{total_imagenes}")

elif seccion == "🎯 Análisis por Plan":
    st.header("🎯 Análisis Detallado por Plan")
    
    # Selector de plan
    plan_seleccionado = st.selectbox(
        "Selecciona un plan para analizar:",
        list(hipotesis_data.keys()),
        format_func=lambda x: f"{x} - {hipotesis_data[x]['nombre']}"
    )
    
    data = hipotesis_data[plan_seleccionado]
    
    # Información del plan
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📊 {data['nombre']}")
        st.markdown(f"**Correlación:** `{data['correlacion']:.3f}`")
        st.markdown(f"**Veredicto:** {data['veredicto']}")
        st.markdown(f"**Insights Clave:** {data['insights']}")
    
    with col2:
        st.metric("Correlación", f"{data['correlacion']:.3f}")
    
    with col3:
        st.metric("Visualizaciones", len(data['imagenes']))
    
    st.markdown("---")
    
    # Galería de imágenes
    st.subheader(f"🖼️ Galería de Visualizaciones - {plan_seleccionado}")
    
    # Selector de imagen
    imagen_seleccionada = st.selectbox(
        "Selecciona una visualización:",
        list(zip(data["descripcion"], data["imagenes"])),
        format_func=lambda x: x[0]
    )
    
    # Mostrar imagen
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

elif seccion == "📈 Análisis Comparativo":
    st.header("📈 Análisis Comparativo de Todas las Variables")
    
    # Tabla comparativa completa
    st.subheader("🔍 Comparación Detallada")
    
    comparativa_data = []
    for plan, data in hipotesis_data.items():
        comparativa_data.append({
            "Plan": plan,
            "Hipótesis": data["nombre"],
            "Correlación": data["correlacion"],
            "Veredicto": data["veredicto"],
            "Tipo": "Positiva" if data["correlacion"] > 0 else "Negativa",
            "Fuerza": "Fuerte" if abs(data["correlacion"]) > 0.5 else "Moderada" if abs(data["correlacion"]) > 0.3 else "Débil" if abs(data["correlacion"]) > 0.1 else "Muy Débil",
            "Visualizaciones": len(data["imagenes"])
        })
    
    df_comparativa = pd.DataFrame(comparativa_data)
    st.dataframe(df_comparativa, use_container_width=True, hide_index=True)
    
    # Gráfico de barras comparativo
    st.subheader("📊 Comparación de Correlaciones")
    
    fig_comp = go.Figure()
    
    for plan, data in hipotesis_data.items():
        color = 'green' if data["correlacion"] > 0 else 'red'
        fig_comp.add_trace(go.Bar(
            name=data["nombre"],
            x=[plan],
            y=[data["correlacion"]],
            marker_color=color,
            text=[f"{data['correlacion']:.3f}"],
            textposition='auto',
        ))
    
    fig_comp.update_layout(
        title="Correlaciones de Todas las Hipótesis",
        xaxis_title="Plan",
        yaxis_title="Correlación con Task Success",
        height=500
    )
    
    # Línea en y=0
    fig_comp.add_hline(y=0, line_dash="dash", line_color="black")
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Análisis de fuerza
    st.subheader("💪 Análisis de Fuerza de Correlaciones")
    
    fuerza_data = []
    for plan, data in hipotesis_data.items():
        fuerza = abs(data["correlacion"])
        if fuerza > 0.5:
            categoria = "Fuerte"
        elif fuerza > 0.3:
            categoria = "Moderada"
        elif fuerza > 0.1:
            categoria = "Débil"
        else:
            categoria = "Muy Débil"
        
        fuerza_data.append({
            "Plan": plan,
            "Hipótesis": data["nombre"],
            "Fuerza Absoluta": fuerza,
            "Categoría": categoria,
            "Dirección": "Positiva" if data["correlacion"] > 0 else "Negativa"
        })
    
    df_fuerza = pd.DataFrame(fuerza_data)
    
    fig_fuerza = px.bar(
        df_fuerza,
        x="Plan",
        y="Fuerza Absoluta",
        color="Dirección",
        title="Fuerza de Correlación por Plan",
        text="Fuerza Absoluta",
        hover_data=["Hipótesis", "Categoría"]
    )
    
    fig_fuerza.update_layout(height=500)
    st.plotly_chart(fig_fuerza, use_container_width=True)

elif seccion == "🔍 Dataset Explorer":
    st.header("🔍 Dataset Explorer")
    
    # Cargar dataset
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
            success_rate = df['task_success'].mean() * 100
            st.metric("Tasa de Éxito", f"{success_rate:.1f}%")
        with col4:
            st.metric("Variables Analizadas", "5")
        
        # Vista previa del dataset
        st.subheader("👁️ Vista Previa del Dataset")
        
        # Mostrar primeras filas
        st.dataframe(df.head(10), use_container_width=True)
        
        # Filtros interactivos
        st.subheader("🔍 Filtros Interactivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtro por éxito
            exito_filtro = st.selectbox(
                "Filtrar por Task Success:",
                ["Todos", "Éxito (1)", "Fracaso (0)"]
            )
            
            if exito_filtro == "Éxito (1)":
                df_filtrado = df[df['task_success'] == 1]
            elif exito_filtro == "Fracaso (0)":
                df_filtrado = df[df['task_success'] == 0]
            else:
                df_filtrado = df
        
        with col2:
            # Filtro por rango de horas de código
            horas_min, horas_max = st.slider(
                "Rango de Horas de Código:",
                float(df['hours_coding'].min()),
                float(df['hours_coding'].max()),
                (float(df['hours_coding'].min()), float(df['hours_coding'].max()))
            )
            
            df_filtrado = df_filtrado[
                (df_filtrado['hours_coding'] >= horas_min) & 
                (df_filtrado['hours_coding'] <= horas_max)
            ]
        
        # Estadísticas del dataset filtrado
        st.subheader(f"📈 Estadísticas del Dataset Filtrado ({len(df_filtrado)} registros)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Registros Filtrados", len(df_filtrado))
        with col2:
            if len(df_filtrado) > 0:
                success_filtrado = df_filtrado['task_success'].mean() * 100
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
        
        # Distribuciones
        st.subheader("📊 Distribuciones de Variables Clave")
        
        variable_dist = st.selectbox(
            "Selecciona variable para visualizar:",
            ["hours_coding", "coffee_intake_mg", "sleep_hours", "cognitive_load", "bugs_reported"]
        )
        
        fig_dist = px.histogram(
            df_filtrado if len(df_filtrado) > 0 else df,
            x=variable_dist,
            color="task_success",
            title=f"Distribución de {variable_dist}",
            barmode="overlay"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error al cargar el dataset: {e}")

elif seccion == "📋 Información del Proyecto":
    st.header("📋 Información del Proyecto")
    
    # Información general
    st.subheader("🎯 Objetivo del Proyecto")
    st.markdown("""
    Analizar el impacto de diferentes factores en la productividad y éxito de los desarrolladores 
    utilizando técnicas de Big Data con PySpark y visualizaciones interactivas.
    """)
    
    # Metodología
    st.subheader("🔬 Metodología")
    st.markdown("""
    - **Tecnología:** PySpark + Pandas + Streamlit
    - **Dataset:** 500 registros de productividad de desarrolladores
    - **Análisis:** Correlación de Pearson + análisis por rangos
    - **Visualizaciones:** 19 gráficos generados con Matplotlib/Seaborn
    """)
    
    # Stack tecnológico
    st.subheader("🛠️ Stack Tecnológico")
    
    tech_data = {
        "Tecnología": ["Python", "PySpark", "Pandas", "Streamlit", "Matplotlib", "Seaborn"],
        "Versión": ["3.14", "3.5.0", "2.3.3", "1.55.0", "3.10.8", "0.13.2"],
        "Propósito": ["Lenguaje principal", "Procesamiento Big Data", "Manipulación datos", "Dashboard interactivo", "Visualización", "Visualización estadística"]
    }
    
    df_tech = pd.DataFrame(tech_data)
    st.dataframe(df_tech, use_container_width=True, hide_index=True)
    
    # Estructura del proyecto
    st.subheader("📁 Estructura del Proyecto")
    
    st.markdown("""
    ```
    BigDataProject/
    ├── data/
    │   └── ai_dev_productivity.csv
    ├── notebooks/
    │   ├── planX_*_analysis.py (5 scripts)
    │   └── results/
    │       ├── plan1-cafeina/
    │       ├── plan2-horas-codigo/
    │       ├── plan3-carga-cognitiva/
    │       ├── plan4-bugs-reportados/
    │       └── plan5-sueno/
    ├── dashboard/
    │   ├── dashboard.py
    │   ├── assets/ (19 imágenes)
    │   └── data/
    ├── docs/
    │   ├── SPECS.md
    │   ├── README.md
    │   ├── CHANGELOG.md
    │   └── ARCHITECTURE.md
    └── .windsurf/
        └── plans/ (5 planes de análisis)
    ```
    """)
    
    # Estándar de calidad
    st.subheader("⭐ Estándar de Calidad")
    
    st.markdown("""
    **Requisitos cumplidos para todos los planes:**
    - ✅ 8 secciones en archivos de estadísticas
    - ✅ 4 requisitos mínimos para gráficos
    - ✅ Formato estandarizado de archivos
    - ✅ Documentación completa y comprensible
    """)
    
    # Conclusiones
    st.subheader("🎉 Conclusiones del Proyecto")
    
    st.markdown("""
    **Hallazgos Principales:**
    1. **Cafeína** es el factor más influyente (r=0.695)
    2. **Horas de código** tienen fuerte correlación positiva (r=0.616)
    3. **Carga cognitiva** impacta negativamente pero débilmente (r=-0.200)
    4. **Bugs reportados** tienen impacto mínimo (r=-0.178)
    5. **Sueño** muestra correlación positiva débil (r=0.187)
    
    **Impacto del Proyecto:**
    - 5 hipótesis analizadas y validadas
    - 19 visualizaciones generadas
    - 1 dashboard interactivo para exploración
    - Evidencia cuantitativa para optimizar productividad
    """)

# Footer
st.markdown("---")
st.markdown("🚀 **BigData Project Dashboard** | Desarrollado con Streamlit | PySpark + Pandas + Python")

# Información de ejecución
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.markdown(f"**Total de imágenes:** {sum(len(data['imagenes']) for data in hipotesis_data.values())}")
    st.markdown(f"**Planes analizados:** {len(hipotesis_data)}")
    st.markdown("**Status:** ✅ Todos los planes completados")
    
    if st.button("🔄 Recargar Dashboard"):
        st.rerun()
