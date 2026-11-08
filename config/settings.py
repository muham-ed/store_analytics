import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "StoreNightAnalytics"
    WORKER_ID: str = Field(default_factory=lambda: f"worker_{os.getenv('HOSTNAME', 'local')}_{os.getpid()}")

    # Database Configuration (PostgreSQL)
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5433
    POSTGRES_DB: str = "store_analytics"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # RF-DETR Model Settings
    RFDETR_MODEL_VARIANT: str = "RFDETRLarge"  # RFDETRBase or RFDETRLarge from rfdetr
    RF_DETR_WEIGHTS_PATH: str = "weights/rfdetr_large.pt"
    RFDETR_CONFIDENCE_THRESHOLD: float = 0.35
    RFDETR_IOU_THRESHOLD: float = 0.50
    DEVICE: str = "cuda"  # NVIDIA GPU Quadro RTX 6000

    # Target Class ID (1 for Person in RF-DETR)
    PERSON_CLASS_ID: int = 1

    # Video Processing Configuration
    FRAME_SKIP: int = 1
    LINE_CROSSING_MIN_FRAMES: int = 5
    HEATMAPS_DIR: str = "outputs/heatmaps"
    LOGS_DIR: str = "logs"

    # Distributed Worker & Resilience Configuration
    POLL_INTERVAL_SECONDS: float = 5.0
    HEARTBEAT_INTERVAL_SECONDS: float = 30.0
    REAPER_INTERVAL_SECONDS: float = 60.0
    TASK_TIMEOUT_SECONDS: float = 300.0
    MAX_ATTEMPTS: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
