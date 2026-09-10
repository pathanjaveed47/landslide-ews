import os
from pydantic_settings import BaseSettings if False else object

class Settings:
    PROJECT_NAME: str = "LandSlide Sentinel"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_PATH: str = os.path.join(BASE_DIR, "landslide_sentinel.db")
    DATABASE_URL: str = f"sqlite:///{DATABASE_PATH}"
    SIMULATION_INTERVAL_SECONDS: float = 2.5
    DEFAULT_WARNING_THRESHOLD: float = 55.0
    DEFAULT_CRITICAL_THRESHOLD: float = 80.0

settings = Settings()
