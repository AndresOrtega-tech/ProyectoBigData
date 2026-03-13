# BigDataProject

Proyecto de análisis de datos enfocado en estudiar la productividad de desarrolladores que utilizan herramientas de Inteligencia Artificial.

## Descripción

Este proyecto realiza un análisis exploratorio de datos (EDA) sobre un dataset que contiene información sobre hábitos de trabajo, consumo de cafeína, horas de sueño, uso de IA y métricas de productividad de desarrolladores. El objetivo principal es identificar patrones y correlaciones que puedan explicar los factores que influyen en el éxito de las tareas de desarrollo.

## Inicio Rápido

### Prerrequisitos
- Python 3.14 o superior
- Java 11 (requerido para PySpark)
- pip (gestor de paquetes de Python)

### Instalación

1. Clona el repositorio (o descarga los archivos)
2. Crea y activa el entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   pip install pyspark==3.5.0
   ```
4. Verifica la instalación de PySpark:
   ```bash
   python -c "from pyspark.sql import SparkSession; print('PySpark funcionando')"
   ```
5. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
6. Abre `notebooks/01_eda_inicial.ipynb` para comenzar el análisis

## Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `jupyter notebook` | Inicia el servidor de Jupyter para análisis interactivo |
| `jupyter lab` | Inicia JupyterLab (interfaz más moderna) |
| `python notebooks/plan1_cafeina_analysis.py` | Ejecuta análisis completo del Plan 1 (Cafeína) |

## Estructura del Proyecto

```
BigDataProject/
├── data/
│   └── ai_dev_productivity.csv    # Dataset principal
├── notebooks/
│   ├── 01_eda_inicial.ipynb       # Análisis exploratorio
│   ├── plan1_cafeina_analysis.py  # Script análisis Plan 1
│   └── results/                   # Resultados organizados por plan
│       └── plan1-cafeina/         # Resultados Plan 1
├── docs/
│   ├── SPECS.md                   # Especificaciones técnicas
│   ├── ARCHITECTURE.md            # Arquitectura del sistema
│   └── CHANGELOG.md               # Registro de cambios
├── .windsurf/
│   └── plans/                     # Planes de análisis estructurados
├── requirements.txt               # Dependencias Python
└── README.md                      # Este archivo
```

## Tecnologías

- **Python 3.14**: Lenguaje principal
- **PySpark 3.5.0**: Procesamiento Big Data local
- **Java 11**: Runtime para PySpark
- **Pandas 3.0.0**: Manipulación y análisis de datos
- **NumPy 2.4.2**: Computación numérica
- **Jupyter**: Entorno de notebooks interactivo
- **Matplotlib 3.10.8**: Visualización de datos
- **Seaborn 0.13.2**: Visualización estadística

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [SPECS.md](docs/SPECS.md) | Especificaciones técnicas detalladas |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitectura y diseño del sistema |
| [CHANGELOG.md](docs/CHANGELOG.md) | Registro de cambios del proyecto |

## Planes de Análisis

El proyecto incluye 6 planes estructurados en `.windsurf/plans/`:

- **Plan 0**: Configuración PySpark local - ✅ Completado
- **Plan 1**: Hipótesis Cafeína vs Task Success - ✅ Completado (correlación: 0.695)
- **Plan 2**: Hipótesis Horas de Código vs Task Success - ✅ Completado (correlación: 0.616)
- **Plan 3**: Hipótesis Carga Cognitiva vs Task Success - ✅ Completado (correlación: -0.200)
- **Plan 4**: Hipótesis Bugs Reportados vs Task Success - ✅ Completado (correlación: -0.178)
- **Plan 5**: Hipótesis Sueño vs Task Success - ✅ Completado (correlación: 0.187)

Cada plan incluye código PySpark, visualizaciones y criterios de validación. Los resultados se guardan en `notebooks/results/{plan-nombre}/`.

## Estándar de Calidad

**Requisito obligatorio para todos los planes**:
- ✅ 8 secciones en archivos de estadísticas (metodología, rangos, insights, etc.)
- ✅ 4 requisitos mínimos para gráficos (títulos, anotaciones, referencias)
- ✅ Formato estandarizado de archivos y carpetas
- 📋 Memory `bigdata-analysis-standards` con referencia completa

**Regla**: Sin excepción, un plan no está "completado" hasta cumplir todos los estándares.

### Resultados Actuales

**Plan 1 (Cafeína)**: ✅ Hipótesis CONFIRMADA
- Correlación: 0.695 (esperada: +0.70)
- Archivos: `notebooks/results/plan1-cafeina/`
- Insights: Mayor consumo de cafeína (>400mg) asociado con 83.6% de éxito

**Plan 2 (Horas de Código)**: ✅ Hipótesis CONFIRMADA
- Correlación: 0.616 (esperada: +0.62)
- Archivos: `notebooks/results/plan2-horas-codigo/`
- Insights: 6-9 horas de código asociado con 85.7% de éxito, rendimientos decrecientes >9h

**Plan 3 (Carga Cognitiva)**: ✅ Hipótesis CONFIRMADA (débil)
- Correlación: -0.200 (esperada: -0.20)
- Archivos: `notebooks/results/plan3-carga-cognitiva/`
- Insights: Carga alta reduce éxito (50.3%), sueño es factor clave (r=-0.734)

**Plan 4 (Bugs Reportados)**: ✅ Hipótesis CONFIRMADA (muy débil)
- Correlación: -0.178 (esperada: -0.18)
- Archivos: `notebooks/results/plan4-bugs-reportados/`
- Insights: 52.2% sesiones sin bugs, umbral crítico 4+ bugs = 0% éxito

**Plan 5 (Sueño)**: ✅ Hipótesis CONFIRMADA (débil)
- Correlación: 0.187 (débil positiva)
- Archivos: `notebooks/results/plan5-sueno/`
- Insights: Punto óptimo 7.1h con 92.3% éxito, déficit severo (<5h) = 18.6% éxito

## Dataset

El dataset `ai_dev_productivity.csv` contiene 500 registros con las siguientes variables:

- `hours_coding`: Horas de programación por sesión
- `coffee_intake_mg`: Miligramos de cafeína consumidos
- `distractions`: Número de distracciones durante el trabajo
- `sleep_hours`: Horas de sueño la noche anterior
- `commits`: Número de commits realizados
- `bugs_reported`: Número de bugs reportados
- `ai_usage_hours`: Horas de uso de herramientas de IA
- `cognitive_load`: Carga cognitiva auto-reportada (1-10)
- `task_success`: Éxito de la tarea (0=fallo, 1=éxito)

## Flujo de Trabajo con IA

Este proyecto utiliza un sistema de documentación automatizada:

- **`/init-docs`**: Genera documentación inicial del proyecto (ejecutado una vez)
- **`/update-docs`**: Actualiza la documentación al final de cada sesión de trabajo

## Contribución

1. Realiza tus análisis en nuevos notebooks dentro de la carpeta `notebooks/`
2. Actualiza la documentación con `/update-docs` al finalizar
3. Mantén el `CHANGELOG.md` actualizado con cambios importantes

## Licencia

<!-- TODO: Agregar licencia -->
