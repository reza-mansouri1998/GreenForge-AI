import sys

def extract_detailed_traceback(error: Exception, error_detail: sys) -> str:
    """
    Extracts the exact file name and line number from the traceback.
    Adopted from standard ML boilerplate, optimized for modern pipelines.
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Script [{file_name}] at line [{line_number}]: {str(error)}"
    
    return str(error)

class GreenForgeBaseError(Exception):
    """Base exception that automatically formats tracebacks for all child errors."""
    def __init__(self, error_message: str):
        # Automatically grab the current sys.exc_info() when raised
        detailed_message = extract_detailed_traceback(error_message, sys)
        super().__init__(detailed_message)
        self.error_message = detailed_message

    def __str__(self):
        return self.error_message

class DataIngestionError(GreenForgeBaseError):
    """Raised when raw data fails to extract, download, or decompress."""
    pass

class DuckDBExecutionError(GreenForgeBaseError):
    """Raised when an out-of-core DuckDB SQL join or transformation fails."""
    pass

class MissingDataError(GreenForgeBaseError):
    """Raised when an expected critical file or table is missing from the data lake."""
    pass