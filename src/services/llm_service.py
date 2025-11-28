# =============================================================================
# FILE: src/services/llm_service.py (FINAL - CLEAN VERSION)
# =============================================================================
"""
LLM service for quiz analysis and code generation using OpenAI API.
"""

from openai import AsyncOpenAI
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential,retry_if_exception_type
from typing import Optional
import re

from ..core.config import settings

logger = structlog.get_logger()


class LLMService:
    """Handles OpenAI LLM interactions with retry logic and structured outputs"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY,timeout=30.0,base_url=settings.OPENAI_BASE_URL)
        self.model = settings.LLM_MODEL
        self.max_completion_tokens  = settings.LLM_MAX_COMPLETIONS_TOKENS
        
        logger.info(
            "llm_service_initialized",
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
        )
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def analyze(self, prompt: str, temperature: float = 1.0) -> str:
        """
        General purpose LLM analysis for understanding requirements.
        Used for: quiz parsing, requirement extraction, decision making.
        
        Args:
            prompt: Analysis request
            temperature: Sampling temperature (1.0)
            
        Returns:
            LLM response text
        """
        logger.info("llm_analyze", prompt_length=len(prompt))
        
        return await self._call_llm(
            prompt=prompt,
            system_message="""You are an expert data analysis task parser. 
            Analyze quiz questions and extract structured information precisely.
            Be thorough and extract ALL relevant details.
            AVAILABLE TOOLS:
            - pandas (as pd)
            - numpy (as np)
            - AnalysisHelpers (helper class)
            AnalysisHelpers METHODS:
            1. preprocess_dataframe(df: pd.DataFrame, safe_mode: bool = True) -> pd.DataFrame
            2. calculate_statistics(df: pd.DataFrame, column: str) -> dict
            
            RULES:
            1. Use `AnalysisHelpers.preprocess_dataframe(df: pd.DataFrame, safe_mode: bool = True) -> pd.DataFrame` immediately after loading data.
            2. ALWAYS print the result or assign it to a variable named `result`.
            3. Do NOT import unsupported libraries.
            4. Handle edge cases (empty data, missing columns).
            """,
            
            # temperature=temperature
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_code(self, prompt: str, data_context: Optional[dict] = None) -> str:
        """
        Generate executable Python code from requirements.
        Used for: data processing, analysis, calculations.
        
        Args:
            prompt: Code generation requirements
            
        Returns:
            Executable Python code (no markdown)
        """
        logger.info("llm_generate_code", prompt_length=len(prompt))
        
        # Ensure data_context is defined for insertion into system message
        data_context = data_context or {}
        
        code = await self._call_llm(
            prompt=prompt,
            system_message="""You are an expert Python Data Analyst.
        
        CRITICAL INSTRUCTIONS:
        1. Use `AnalysisHelpers.preprocess_dataframe(df: pd.DataFrame, safe_mode: bool = True) -> pd.DataFrame` to clean data.
        2. The data is ALREADY LOADED in the `data` dictionary.
        3. ACCESS IT LIKE THIS: `df = data['source_0']['parsed']['dataframe']`
        4. Do NOT load files with pd.read_csv() again.
        5. Check the 'candidates' provided in context for potential answers (sum, min, max).
        6. If the question implies a 'cutoff' or 'threshold', look for that SPECIFIC value.
        
        Context:
        """ + str(data_context)
            ,
            # temperature=0.2
        )
        
        # Clean up any markdown artifacts
        return self._clean_code(code)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,

    )
    async def solve(self, prompt: str, temperature: float = 1.0) -> str:
        """
        Direct problem solving without code generation.
        Used for: simple calculations, direct answers, text analysis.
        
        Args:
            prompt: Problem to solve
            temperature: Sampling temperature
            
        Returns:
            Direct answer
        """
        logger.info("llm_solve", prompt_length=len(prompt))
        
        return await self._call_llm(
            prompt=prompt,
            system_message="""You are a precise problem solver.
Provide ONLY the final answer without explanations unless requested.
Be accurate and concise.""",
            # temperature=temperature
        )
        async def analyze_json(self, prompt: str) -> dict:
            """Expects JSON output"""
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a parser. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
        return json.loads(response.choices[0].message.content)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 1.0
    ) -> str:
        """
        Internal method to call OpenAI API with consistent error handling.
        
        Args:
            prompt: User prompt
            system_message: System instruction
            temperature: Sampling temperature
            
        Returns:
            LLM response content
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                # temperature=temperature,
                max_completion_tokens =self.max_completion_tokens 
            )
            
            result = response.choices[0].message.content
            
            logger.info(
                "llm_call_success",
                model=self.model,
                response_length=len(result),
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
            return result
            
        except Exception as e:
            message = str(e)
            short_msg = message[:500]
            logger.error(
                "llm_call_error",
                model=self.model,
                error=short_msg,
                exc_info=True,
            )
            raise
    @staticmethod
    def _clean_code(code: str) -> str:
        """
        Remove markdown formatting from generated code.
        
        Args:
            code: Raw LLM output
            
        Returns:
            Clean Python code
        """
        # Remove markdown code fences like ```python ... ``` or ``` ... ```
        # Use regex to strip leading/trailing triple-backtick fences and optional language
        code = re.sub(r'^\s*```(?:\w+)?\s*', '', code)
        code = re.sub(r'\s*```\s*$', '', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code
        # Remove leading/trailing whitespac
        def estimate_tokens(self, text: str) -> int:
            return len(text) // 4