# 🛡️ Alerta Raven - Sistema de Emergencias Inteligente

**Alerta Raven** es un sistema integral que combina una app móvil con sensores inteligentes y un backend basado en inteligencia artificial para brindar una respuesta inmediata ante emergencias como caídas, accidentes vehiculares o situaciones críticas detectadas manualmente o automáticamente.

---

## 🚀 Características Principales

- 🚨 Detección en tiempo real de eventos de emergencia
- 📱 Integración con sensores (Acelerómetro, Giroscopio, Magnetómetro)
- 📍 Geolocalización automática con mapas
- 🤖 Clasificación de eventos con modelo de Machine Learning
- 🧠 Auto-aprendizaje periódico para mejorar precisión
- 📊 Panel de administración para gestión y monitoreo de alertas
- 📲 Envío de SMS y notificaciones automáticas
- 🎞️ Animaciones interactivas con Lottie

---

## 🧠 Componentes del Proyecto

### App Móvil (React Native + Expo)
- Detección de sacudidas
- Botón de pánico físico (BackHandler)
- Llamada automática al 911
- Visualización de estado del sistema
- Contactos de emergencia sincronizados

### Backend (FastAPI + MongoDB)
- API REST para recibir alertas
- Clasificación automática con Random Forest
- Almacenamiento en MongoDB
- Entrenamiento continuo de modelos
- Logs y analíticas de eventos

---

## ⚙️ Requisitos Técnicos

### Backend
- Python 3.9+
- MongoDB 4.4+
- FastAPI
- Uvicorn

### App Móvil
- Node.js 16+
- Expo CLI
- React Native

---

## 🧪 Instalación del Backend

```bash
git clone https://github.com/Al3x2023/ApiRaven.git
cd alerta-raven
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

### Crear `.env` con configuración:

```ini
MONGODB_URI=mongodb://localhost:27017/alerta_raven
API_SECRET_KEY=clave-secreta-123
```

### Iniciar servidor:
```bash
uvicorn main:app --reload
```

Accede al backend en: `http://localhost:8000`

---

## 📱 Instalación de la App Móvil

```bash
cd app-mobile
npm install
npx expo start
```

> Asegúrate de tener habilitados permisos de sensores, ubicación, y contactos.

---

## 🔁 Estructura del Proyecto

```
alerta-raven/
├── api/
│   ├── endpoints/
│   ├── models/
│   └── utils/
├── core/
│   ├── detection/
│   └── ml/
├── static/
├── app-mobile/               # App React Native
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🤖 Modelo de Machine Learning

- Algoritmo: Random Forest
- Entrenado con datos de acelerómetro, giroscopio, y magnitud del vector
- Clasificación en 5 tipos de eventos:
  - `normal`
  - `phone_drop`
  - `vehicle_accident`
  - `manual`
  - `other_impact`

---

## 🐳 Despliegue en Producción

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

**O con Docker:**

```bash
docker-compose up -d
```

---

## 🧩 Endpoints Clave

- `POST /emergency/sensor-data` - Recibe y clasifica datos en tiempo real
- `POST /emergency/manual-alert` - Activa alerta manual
- `GET /emergency/active-alerts` - Lista alertas recientes

---

## 🤝 Contribuciones

1. Haz un fork
2. Crea tu rama (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit (`git commit -m 'Agrega nueva funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request 🚀

---

## 📄 Licencia

Distribuido bajo la licencia **MIT**. Ver el archivo `LICENSE` para más información.

---

## 📬 Contacto

**Equipo de Desarrollo Alerta Raven**  
📧 desarrollo@alerta-raven.com  
🔗 [github.com/Al3x2023](https://github.com/Al3x2023)

---

*Documentación generada el 27 de junio de 2025.*