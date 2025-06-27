from pathlib import Path

class Settings:
    APP_NAME = "Alerta Raven"
    STATIC_DIR = Path(__file__).parent.parent / "static"
    
settings = Settings()