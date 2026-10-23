"""AI Data Analyst: natural-language analytics over CSV files."""
from .config import Settings
from .session import AnalystSession

__all__ = ["AnalystSession", "Settings"]
__version__ = "1.0.0"
