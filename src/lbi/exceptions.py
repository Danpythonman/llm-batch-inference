"""Custom exceptions for the llmbatch package."""

from __future__ import annotations

# --- Public API ---

__all__: list[str] = [
    'BatchCancelledError',
    'BatchCreationError',
    'BatchLLMError',
    'BatchNotFoundError',
    'BatchPollTimeoutError',
    'ProviderError',
    'ResultsNotReadyError',
]


# --- Classes ---


class BatchLLMError(Exception):
    """Base exception for all llmbatch errors."""


class BatchCreationError(BatchLLMError):
    """Raised when batch creation fails at the provider."""


class BatchNotFoundError(BatchLLMError):
    """Raised when a batch ID cannot be found."""


class BatchPollTimeoutError(BatchLLMError):
    """Raised when polling exceeds the configured timeout."""


class BatchCancelledError(BatchLLMError):
    """Raised when operating on a batch that was cancelled."""


class ProviderError(BatchLLMError):
    """Raised when a provider SDK returns an unexpected error."""

    def __init__(
        self,
        provider: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.cause = cause
        super().__init__(f'[{provider}] {message}')


class ResultsNotReadyError(BatchLLMError):
    """Raised when results are requested before the batch ends."""
