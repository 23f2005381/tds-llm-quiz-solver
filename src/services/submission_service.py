# =============================================================================
# FILE: src/services/submission_service.py
# =============================================================================
import aiohttp
import structlog
import json
from typing import Any, Dict

logger = structlog.get_logger()


class SubmissionService:
    """Handles answer submission to quiz endpoints"""
    
    async def submit_answer(
        self,
        email: str,
        secret: str,
        url: str,
        answer: Any,
        submit_url: str = None
    ) -> Dict[str, Any]:
        """Submit answer to quiz endpoint"""
        
        # Use provided submit_url or derive from url
        if not submit_url:
            # Default submission endpoint
            submit_url = url.replace('/quiz-', '/submit-') if '/quiz-' in url else url
        
        payload = {
            "email": email,
            "secret": secret,
            "url": url,
            "answer": answer
        }
        
        logger.info("submitting_answer", submit_url=submit_url, answer_type=type(answer).__name__)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                submit_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    result = await response.json()
                else:
                    # Not JSON - read as text
                    text = await response.text()
                    logger.warning(
                        "submit_returned_non_json",
                        content_type=content_type,
                        text=text[:200]
                    )
                    # Try to parse anyway
                    try:
                        result = json.loads(text)
                    except:
                        # Return error structure
                        result = {
                            "correct": False,
                            "reason": f"Server returned non-JSON: {text[:100]}",
                            "url": None
                        }
                logger.info(
                    "submission_result",
                    correct=result.get("correct"),
                    has_next_url=bool(result.get("url"))
                )

                return result