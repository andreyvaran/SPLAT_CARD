from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="S3__",
    )
    service_name: str
    endpoint_url: str
    region_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    addressing_style: str
    global_endpoint_url: str
    path: str
    file_path: str


s3_settings = S3Settings()
