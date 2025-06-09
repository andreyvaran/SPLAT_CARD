from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class FileStorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="S3__",
    )
    service_name: str = "FileStorage"
    root_path: str = "results/"


file_settings = FileStorageSettings()
