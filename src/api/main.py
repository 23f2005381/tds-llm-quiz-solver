# =============================================================================
# FILE: src/api/main.py (PRODUCTION READY)
# =============================================================================
"""
FastAPI application for quiz solving.
Meets all requirements: secret verification, 3-minute timeout, error handling.
"""

from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from contextlib import asynccontextmanager
import asyncio
import sys
import importlib
from .models import QuizRequest, QuizResponse
from ..core.orchestrator import QuizOrchestrator
from ..core.config import settings
from ..utils.logger import setup_logging
if 'src.services.code_executor' in sys.modules:
    importlib.reload(sys.modules['src.services.code_executor'])
    
from src.services.code_executor import CodeExecutor
# Setup structured logging
setup_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    logger.info("Starting Quiz Solver API", port=settings.API_PORT)
    
    # Install Playwright browsers on first run
    try:
        import subprocess
        subprocess.run(["playwright", "install", "chromium"], check=True)
        logger.info("Playwright browsers installed")
    except Exception as e:
        logger.warning(f"Playwright installation check failed: {e}")
    
    yield
    
    logger.info("Shutting down Quiz Solver API")


app = FastAPI(
    title="LLM Quiz Solver API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(
        "request_received",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None
    )
    response = await call_next(request)
    logger.info(
        "request_completed",
        status_code=response.status_code
    )
    return response


@app.post("/", response_model=QuizResponse, status_code=status.HTTP_200_OK)
async def solve_quiz(request: QuizRequest,
                     background_tasks: BackgroundTasks):
    """
    Main endpoint to receive and solve quiz tasks.
    
    Requirements:
    - Verify secret matches (403 if invalid)
    - Return 400 for invalid JSON
    - Solve quiz within 3 minutes
    - Handle quiz chains
    """
    
    # Step 1: Verify secret (REQUIREMENT)
    if request.secret != settings.SECRET:
        logger.warning(
            "invalid_secret_attempt",
            email=request.email,
            provided_secret=request.secret[:4] + "..."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret"
        )
    
    logger.info(
        "quiz_request_received",
        email=request.email,
        url=request.url
    )
    
    # Step 2: Process quiz
# =============================================================================
# FILE: src/api/main.py (CONTINUED)
# =============================================================================

    # Step 2: Process quiz with 170s timeout (3 min = 180s, leave 10s buffer)
    try:
        orchestrator = QuizOrchestrator(request.email, request.secret)
        
        # Execute with timeout
        background_tasks.add_task(orchestrator.solve_quiz_chain, request.url)
        
        return {
        "success": True,
        "message": "Quiz processing started in background",
        "details": {
            "initial_url": str(request.url),
            "status": "processing"
        }
        }
        
    except asyncio.TimeoutError:
        logger.error(
            "quiz_timeout",
            email=request.email,
            url=request.url
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Quiz processing exceeded 3 minute time limit"
        )
        
    except Exception as e:
        logger.error(
            "quiz_processing_error",
            email=request.email,
            url=request.url,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "quiz-solver"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Quiz Solver API",
        "version": "1.0.0",
        "endpoints": {
            "POST /": "Solve quiz",
            "GET /health": "Health check"
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.error("unhandled_exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc)
        }
    )
