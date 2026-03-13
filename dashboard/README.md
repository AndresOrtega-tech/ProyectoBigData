# BigData Project - Streamlit Dashboard

## 🚀 Deploy Automático

### Opción 1: Streamlit Cloud (Recomendado - Gratis)

1. **Subir a GitHub**:
```bash
git init
git add .
git commit -m "Dashboard Streamlit completo - BigData Project"
git branch -M main
git remote add origin https://github.com/tu-usuario/bigdata-project.git
git push -u origin main
```

2. **Deploy en Streamlit Cloud**:
- Ve a: https://share.streamlit.io
- Click "New app" → "Connect GitHub"
- Selecciona tu repositorio
- Main file: `dashboard/dashboard.py`
- Click "Deploy"

### Opción 2: Railway (Gratis)

1. **Instalar Railway CLI**:
```bash
npm install -g @railway/cli
```

2. **Crear `railway.json`**:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run dashboard/dashboard.py --server.port=$PORT --server.address=0.0.0.0",
    "healthcheckPath": "/"
  }
}
```

3. **Deploy**:
```bash
railway login
railway up
```

## 📋 Estructura para Deploy

```
BigDataProject/
├── dashboard/
│   ├── dashboard.py          # Archivo principal
│   ├── assets/              # 19 imágenes
│   └── data/
│       └── ai_dev_productivity.csv
├── requirements.txt         # Dependencias
└── README.md
```

## ⚙️ Configuración

### Variables de Entorno (No necesarias para este proyecto)
- No se requieren variables especiales
- Todo funciona con archivos locales

### Archivos Clave
- `dashboard/dashboard.py` - Código del dashboard
- `requirements.txt` - Dependencias actualizadas
- `dashboard/assets/` - Todas las visualizaciones

## 🔧 Ejecutar Localmente

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
cd dashboard
streamlit run dashboard.py
```

## 🌐 URLs de Ejemplo

- **Streamlit Cloud**: `https://tu-username-bigdata-project.streamlit.app`
- **Railway**: `https://tu-app.railway.app`
- **Vercel**: `https://tu-app.vercel.app`

## 📊 Características del Dashboard

- ✅ 19 visualizaciones integradas
- ✅ 5 planes de análisis completos
- ✅ Navegación interactiva
- ✅ Dataset explorer con filtros
- ✅ Análisis comparativo en tiempo real
- ✅ Diseño responsive y profesional

## 🎯 Ventajas del Deploy

1. **Acceso 24/7**: Dashboard siempre disponible
2. **URL pública**: Compartir con stakeholders
3. **Actualizaciones automáticas**: Con cada push a GitHub
4. **Gratis**: Opciones sin costo para proyectos personales
5. **Escalable**: Soporta múltiples usuarios simultáneos

---

**Listo para deploy!** 🚀
