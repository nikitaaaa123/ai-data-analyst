"""Domain exceptions. Each carries a message that is safe to show to end users."""


class AnalystError(Exception):
    """Base class for all application errors."""


class DataValidationError(AnalystError):
    """Uploaded data is unusable (empty, too big, unparsable...)."""


class UnsafeQueryError(AnalystError):
    """SQL or Python code was rejected by the safety checks."""


class QueryExecutionError(AnalystError):
    """A valid query failed while running."""


class ChartError(AnalystError):
    """A chart spec could not be rendered."""


class LLMError(AnalystError):
    """The LLM provider call failed."""
