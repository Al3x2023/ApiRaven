from pathlib import Path
from fastapi.responses import FileResponse
from core.config import settings

class StaticService:
    @staticmethod
    def serve_html(filename: str):
        file_path = settings.STATIC_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo {filename} no encontrado en {settings.STATIC_DIR}")
        return FileResponse(file_path)