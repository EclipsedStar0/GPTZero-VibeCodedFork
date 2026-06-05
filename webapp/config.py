from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_id: str = "gpt2"
    low_threshold: int = 60
    high_threshold: int = 80
    stride: int = 512
    max_length: Optional[int] = None
    minimum_tokens: int = 32
    char_to_token_ratio: float = 0.3
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    human_text_file: Optional[str] = "calibration_data/HUMAN_TEXT.txt"
    ai_text_file: Optional[str] = "calibration_data/LLM_TEXT.txt"


settings = Settings()