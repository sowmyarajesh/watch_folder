from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    input_folder: Path = Path("data/input")
    results_folder: Path = Path("data/results")
    db_path: str = "DB/app_db.sqlite"

    # Worker config — for large volume, tune to your CPU/GPU core count
    max_worker_threads: int = 4       # thread pool size for YOLOv8 inference
    queue_max_size: int = 500         # bounded queue to cap memory pressure
    yolo_model_name: str = "yolov8n.pt"  # nano=fast; swap to yolov8m.pt for accuracy
    yolo_confidence: float = 0.25

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()

# Ensure directories exist at import time
settings.input_folder.mkdir(parents=True, exist_ok=True)
settings.results_folder.mkdir(parents=True, exist_ok=True)
Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
