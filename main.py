from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import static_endpoints, emergency_endpoints

app = FastAPI(
    title="Alerta Raven",
    description="Sistema de emergencias"
)

# Configuración de CORS
# Puedes ajustar estos valores según tus necesidades
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (en producción, especifica los dominios correctos)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los encabezados
)

# ✅ Monta la carpeta 'static' para servir archivos como HTML, CSS, imágenes, y MP4
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Incluye tus rutas personalizadas
app.include_router(static_endpoints.router)
app.include_router(emergency_endpoints.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)