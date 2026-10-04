# src/greenforge_ai/utils/exceptions.py

from __future__ import annotations

import sys
from types import TracebackType
from typing import Optional, Union


def extract_detailed_traceback(
    error: Union[BaseException, str],
    exc_tb: Optional[TracebackType] = None,
) -> str:
    """
    Return a clear error message with the deepest available source location.

    If called inside an ``except`` block, the active traceback is used
    automatically. A traceback may also be passed explicitly.

    Examples:
        try:
            ...
        except Exception as exc:
            raise DataIngestionError(exc) from exc

        raise MissingDataError("Expected telemetry file is missing")
    """
    message = str(error)

    # Prefer an explicitly supplied traceback.
    tb = exc_tb

    # If an exception object already carries its traceback, use it.
    if tb is None and isinstance(error, BaseException):
        tb = error.__traceback__

    # Fall back to the currently handled exception, if any.
    if tb is None:
        _, _, tb = sys.exc_info()

    if tb is None:
        return message

    # Walk to the deepest frame: this is the line where the failure originated.
    while tb.tb_next is not None:
        tb = tb.tb_next

    frame = tb.tb_frame
    file_name = frame.f_code.co_filename
    function_name = frame.f_code.co_name
    line_number = tb.tb_lineno

    return f"Script [{file_name}] | Function [{function_name}] | Line [{line_number}]: {message}"


class GreenForgeBaseError(Exception):
    """
    Base exception for GreenForge pipeline errors.

    Accepts either a normal message or an existing exception. When an existing
    exception is provided, its traceback location is preserved in the formatted
    message.
    """

    def __init__(self, error: Union[BaseException, str]):
        self.original_error = error if isinstance(error, BaseException) else None
        self.error_message = extract_detailed_traceback(error)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message


class DataIngestionError(GreenForgeBaseError):
    """Raised when raw data fails to extract, download, or decompress."""


class DuckDBExecutionError(GreenForgeBaseError):
    """Raised when an out-of-core DuckDB SQL join or transformation fails."""


class MissingDataError(GreenForgeBaseError):
    """Raised when an expected critical file or table is missing."""


class DataValidationError(GreenForgeBaseError):
    """Raised when processed data violates a required validation contract."""


class DataTransformationError(GreenForgeBaseError):
    """Raised when a transformation or feature-engineering stage fails."""
