from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

@router.get("/")
async def serve_index():
    file_path = Path(__file__).parent.parent.parent / "static" / "index.html"
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró index.html en {file_path.parent}")
    return FileResponse(file_path)

@router.get("/landing")
async def serve_landing():
    file_path = Path(__file__).parent.parent.parent / "static" / "landing.html"
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró landing.html en {file_path.parent}")
    return FileResponse(file_path)