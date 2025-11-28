# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration using Pydantic Settings"""
    
    # API Configuration
    SECRET: str
    EMAIL: str
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-5-nano"
    LLM_TEMPERATURE: float = 1.0
    LLM_MAX_COMPLETIONS_TOKENS: int = Field(default=4096, env="LLM_max_completion_tokens ")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    # Execution Limits
    QUIZ_TIMEOUT_SECONDS: int = 170  # Buffer for 180s limit
    CODE_EXECUTION_TIMEOUT: int = 30
    MAX_ITERATIONS: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Browser Configuration
    BROWSER_HEADLESS: bool = True
    BROWSER_TIMEOUT: int = 30000

    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2  # seconds

    MAX_FILE_SIZE_MB: int = 50
    TEMP_DIR: str = "/tmp/quiz_solver"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields


settings = Settings()
Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)