
# # =============================================================================
# # FILE: src/api/models.py
# # =============================================================================
# from pydantic import BaseModel, EmailStr, Field
# from typing import Optional, Any


# class QuizRequest(BaseModel):
#     email: EmailStr
#     secret: str
#     url: str = Field(..., description="Quiz URL to solve")


# class QuizResponse(BaseModel):
#     success: bool
#     message: str
#     details: Optional[dict] = None


# # =============================================================================
# # FILE: src/core/config.py
# # =============================================================================
# from pydantic_settings import BaseSettings
# from typing import Optional


# class Settings(BaseSettings):
#     # API Configuration
#     SECRET: str
#     EMAIL: str
#     API_HOST: str = "0.0.0.0"
#     API_PORT: int = 8000
    
#     # LLM Configuration
#     OPENAI_API_KEY: str
#     LLM_MODEL: str = "gpt-5-nano"  # Use latest stable model
#     LLM_TEMPERATURE: float = 1.0
#     LLM_max_completion_tokens : int = 4096
    
#     # Browser Configuration
#     BROWSER_HEADLESS: bool = True
#     BROWSER_TIMEOUT: int = 30000
    
#     # Retry Configuration
#     MAX_RETRIES: int = 3
#     RETRY_DELAY: int = 2
    
#     # File Processing
#     MAX_FILE_SIZE_MB: int = 50
#     TEMP_DIR: str = "/tmp/quiz_solver"
    
#     class Config:
#         env_file = ".env"
#         case_sensitive = True


# settings = Settings()
# =============================================================================
# FILE: src/api/models.py
# =============================================================================
"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, EmailStr, Field, HttpUrl
from typing import Optional, Any, Dict


class QuizRequest(BaseModel):
    """
    Quiz request payload (matches instructor's spec).
    """
    email: EmailStr = Field(..., description="Your email address")
    secret: str = Field(..., min_length=1, description="Secret from Google Form")
    url: HttpUrl = Field(..., description="Quiz URL to solve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "student@example.com",
                "secret": "your-secret-key",
                "url": "https://example.com/quiz-834"
            }
        }


class QuizResponse(BaseModel):
    """
    Quiz response payload.
    """
    success: bool = Field(..., description="Whether processing was successful")
    message: str = Field(..., description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Quiz processing completed",
                "details": {
                    "total_quizzes": 3,
                    "success_count": 3,
                    "results": []
                }
            }
        }


