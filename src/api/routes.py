# src/api/routes.py
from fastapi import APIRouter, HTTPException, status, Request
from src.api.models import QuizRequest, QuizResponse
from src.core.orchestrator import QuizOrchestrator
from src.core.config import settings
import structlog
import asyncio

router = APIRouter()
logger = structlog.get_logger()

@router.post("/", response_model=QuizResponse, status_code=status.HTTP_200_OK)
async def solve_quiz(request: QuizRequest, req: Request):
    """
    Main endpoint to receive and process quiz tasks.
    Validates secret, initiates quiz solving process.
    """
    # Validate secret
    if request.secret != settings.SECRET:
        logger.warning("Invalid secret attempt", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret"
        )
    
    logger.info("Quiz request received", email=request.email, url=request.url)
    
    # Process quiz with timeout (170s buffer for 180s limit)
    try:
        orchestrator = QuizOrchestrator(request.email, request.secret)
        async with asyncio.timeout(170):
            result = await orchestrator.solve_quiz_chain(request.url)
        
        return QuizResponse(
            status="success",
            message="Quiz processing completed",
            details=result
        )
    
    except asyncio.TimeoutError:
        logger.error("Quiz timeout", url=request.url)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Processing timeout exceeded"
        )
    except Exception as e:
        logger.exception("Quiz processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )
# src/api/routes.py
from fastapi import APIRouter, HTTPException, status, Request
from src.api.models import QuizRequest, QuizResponse
from src.core.orchestrator import QuizOrchestrator
from src.core.config import settings
import structlog
import asyncio

router = APIRouter()
logger = structlog.get_logger()

@router.post("/", response_model=QuizResponse, status_code=status.HTTP_200_OK)
async def solve_quiz(request: QuizRequest, req: Request):
    """
    Main endpoint to receive and process quiz tasks.
    Validates secret, initiates quiz solving process.
    """
    # Validate secret
    if request.secret != settings.SECRET:
        logger.warning("Invalid secret attempt", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret"
        )
    
    logger.info("Quiz request received", email=request.email, url=request.url)
    
    # Process quiz with timeout (170s buffer for 180s limit)
    try:
        orchestrator = QuizOrchestrator(request.email, request.secret)
        async with asyncio.timeout(170):
            result = await orchestrator.solve_quiz_chain(request.url)
        
        return QuizResponse(
            status="success",
            message="Quiz processing completed",
            details=result
        )
    
    except asyncio.TimeoutError:
        logger.error("Quiz timeout", url=request.url)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Processing timeout exceeded"
        )
    except Exception as e:
        logger.exception("Quiz processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )
