# =============================================================================
# FILE: src/core/exceptions.py
# =============================================================================
class QuizSolverException(Exception):
    """Base exception for quiz solver"""
    pass


class InvalidSecretError(QuizSolverException):
    """Invalid secret provided"""
    pass


class QuizTimeoutError(QuizSolverException):
    """Quiz solving exceeded time limit"""
    pass


class DataFetchError(QuizSolverException):
    """Error fetching data"""
    pass