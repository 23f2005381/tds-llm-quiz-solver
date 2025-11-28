# =============================================================================
# FILE: src/services/code_executor.py (ENHANCED VERSION)
# =============================================================================
"""
Enhanced code executor with timeout, security checks, and custom context support.
Safely executes LLM-generated code with helper functions available.
"""

import ast
import builtins
import structlog
from typing import Any, Dict, Optional
import sys
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import asyncio
import signal
import builtins
from .analysis_helpers import AnalysisHelpers

logger = structlog.get_logger()


class CodeExecutionTimeout(Exception):
    """Raised when code execution times out"""
    pass


class CodeExecutor:
    """
    Safely executes generated code in a controlled environment.
    Supports custom context (AnalysisHelpers), timeouts, and security checks.
    """
    
    def __init__(self, timeout: int = 60):
        """
        Initialize executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.blocked_modules = {
           'os', 'subprocess', 'sys', 'eval', 'exec', 'compile', 'open', 'file',
            'requests', 'aiohttp', 'httpx', 'urllib', 'urllib3', 'socket', 'http'
        }
    
    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        module_root = name.split('.')[0]
        if module_root in self.blocked_modules:
            raise ImportError(f"Import of '{module_root}' blocked for security.")
        
        logger.debug("safe_import_allowed", module=name)
        
        # ✅ Use builtins.__import__ directly - bulletproof
        return builtins.__import__(name, globals or {}, locals or {}, fromlist, level)
    
    async def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code with optional custom context.
        
        Args:
            code: Python code to execute
            context: Custom execution context (data, helpers, etc.)
            
        Returns:
            Dict with result, output, success status
        """
        
        logger.info("executing_code", code_length=len(code))
        
        # Step 1: Validate code syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.error("code_syntax_error", error=str(e))
            return {
                "success": False,
                "error": f"Syntax error: {str(e)}",
                "result": None
            }
        
        # Step 2: Security check
        if not self._is_code_safe(code):
            logger.error("code_security_violation")
            return {
                "success": False,
                "error": "Code contains unsafe operations",
                "result": None
            }
        
        # Step 3: Prepare execution context
        exec_context = self._prepare_context(context)
        
        # Step 4: Execute with timeout
        try:
            result = await asyncio.wait_for(
                self._execute_code(code, exec_context),
                timeout=self.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error("code_execution_timeout", timeout=self.timeout)
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout} seconds",
                "result": None
            }
        except Exception as e:
            logger.error("code_execution_exception", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def _is_code_safe(self, code: str) -> bool:
        """
        Check if code contains unsafe operations.
        Basic security check - blocks dangerous imports and operations.
        """
        
        code_lower = code.lower()
        
        # Check for blocked imports
        for blocked in self.blocked_modules:
            if f'import {blocked}' in code_lower or f'from {blocked}' in code_lower:
                logger.warning(f"Blocked import detected: {blocked}")
                return False
        
        # Check for HTTP libraries
        http_libraries = ['requests', 'aiohttp', 'httpx', 'urllib.request', 'urllib3']
        for lib in http_libraries:
            if f'import {lib}' in code_lower or f'from {lib}' in code_lower:
                logger.warning(f"Blocked HTTP library: {lib}")
                return False

        # Check for dangerous function calls
        dangerous_patterns = [
            'requests.', 'aiohttp.', 'httpx.', 'urllib.', 
            '__import__', 'eval(', 'exec(', 'compile(', 'open(',
            'file(', 'input(', 'raw_input(', 'globals()', 'locals()',
            'session.post', 'session.get', 'requests.post', 'requests.get'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        
        return True
    
    def _prepare_context(self, custom_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare execution context with libraries and custom objects.
        
        Args:
            custom_context: Custom context (data, helpers, etc.)
            
        Returns:
            Complete execution context
        """
        safe_builtins = {
            k: v for k, v in builtins.__dict__.items()
            if k not in ['eval', 'exec', 'compile', 'open', 'file', 'quit', 'exit', 'input']
        }
        
        # ✅ CRITICAL: Add safe import wrapper
        safe_builtins['__import__'] = self._safe_import
        
        # ✅ Base context with standard libraries
        context = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'AnalysisHelpers': AnalysisHelpers,
            'result': None,
            '__builtins__': safe_builtins  # ✅ This must be set BEFORE custom_context
        }
        
        # ✅ FIX: Merge custom_context but PRESERVE __builtins__
        if custom_context:
            # Don't let custom_context override __builtins__
            custom_builtins = custom_context.pop('__builtins__', None)
            context.update(custom_context)
            
            # Restore our safe builtins (don't use custom one)
            context['__builtins__'] = safe_builtins
        
        return context
    
    async def _execute_code(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute code in prepared context.
        Runs in asyncio executor to support timeout.
        """
        
        loop = asyncio.get_event_loop()
        
        # Run in executor (thread pool) to avoid blocking
        result = await loop.run_in_executor(
            None,
            self._run_code_sync,
            code,
            context
        )
        
        return result
    
    def _run_code_sync(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronous code execution with output capture.
        This is called from executor to support async timeout.
        """
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_output = StringIO()
        sys.stderr = captured_errors = StringIO()
        
        try:
            # Execute the code
            exec(code, context)
            
            # ENHANCED: Check if result exists and is not None
            result = context.get('result')
            
            # NEW: If result is None but code printed output, try to extract answer from output
            output = captured_output.getvalue()
            errors = captured_errors.getvalue()
            
            if result is None and output:
                # Try to extract potential answer from the last line of output
                output_lines = output.strip().split('\n')
                if output_lines:
                    last_line = output_lines[-1].strip()
                    # If last line looks like a simple answer (number, simple string)
                    if (last_line and 
                        (last_line.replace('.', '').replace('-', '').isdigit() or 
                         len(last_line) < 100)):  # Not too long
                        logger.info("Extracted result from output", extracted_result=last_line)
                        result = last_line
            
            # Handle matplotlib figures (if any were created)
            figure_base64 = None
            if plt.get_fignums():
                figure_base64 = self._capture_matplotlib_figure()
            
            # If result is None but there's a figure, use the figure
            if result is None and figure_base64:
                result = figure_base64
            
            logger.info(
                "code_executed_successfully",
                has_result=result is not None,
                has_figure=figure_base64 is not None,
                output_length=len(output)
            )
            
            return {
                "success": True,
                "result": result,
                "output": output,
                "errors": errors if errors else None,
                "figure": figure_base64
            }
            
        except Exception as e:
            error_output = captured_output.getvalue()
            error_stderr = captured_errors.getvalue()
            
            logger.error(
                "code_execution_error",
                error=str(e),
                error_type=type(e).__name__,
                output=error_output,
                stderr=error_stderr
            )
            
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "output": error_output,
                "errors": error_stderr,
                "result": None
            }
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Clean up matplotlib
            plt.close('all')
    
    def _capture_matplotlib_figure(self) -> str:
        """
        Capture matplotlib figure as base64-encoded PNG.
        
        Returns:
            Base64-encoded image string with data URI prefix
        """
        try:
            buf = BytesIO()
            plt.savefig(
                buf,
                format='png',
                dpi=150,
                bbox_inches='tight',
                facecolor='white'
            )
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Return as data URI
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.warning(f"Failed to capture matplotlib figure: {e}")
            return None
        finally:
            plt.close('all')
    
    def validate_result(
        self,
        result: Any,
        expected_type: Optional[str] = None
    ) -> bool:
        """
        Validate that result matches expected type.
        
        Args:
            result: Execution result
            expected_type: Expected type ('number', 'string', 'json', etc.)
            
        Returns:
            True if valid
        """
        
        if result is None:
            return False
        
        if expected_type == 'number':
            return isinstance(result, (int, float)) and not isinstance(result, bool)
        
        elif expected_type == 'string':
            return isinstance(result, str)
        
        elif expected_type == 'boolean':
            return isinstance(result, bool)
        
        elif expected_type in ['json', 'object']:
            return isinstance(result, dict)
        
        elif expected_type == 'array':
            return isinstance(result, (list, tuple))
        
        # If no expected type specified, any non-None result is valid
        return True