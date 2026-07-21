"""Shared data models for batch requests, responses, and status."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

__all__: list[str] = [
    'BatchInfo',
    'BatchRequest',
    'BatchResult',
    'BatchResultStatus',
    'BatchStatus',
    'Message',
    'Role',
]


class Role(enum.StrEnum):
    """Message role in a conversation turn."""

    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class BatchStatus(enum.StrEnum):
    """Normalised batch lifecycle status across providers."""

    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    EXPIRED = 'expired'


class BatchResultStatus(enum.StrEnum):
    """Outcome status for an individual request within a batch."""

    SUCCEEDED = 'succeeded'
    ERRORED = 'errored'
    CANCELLED = 'cancelled'
    EXPIRED = 'expired'


# --- Data classes ---


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: Role
    content: str


@dataclass(frozen=True)
class BatchRequest:
    """A single request to include in a batch.

    Each request carries a unique custom_id so results can be
    matched back to inputs regardless of return order.

    Args:
        custom_id: Caller-defined identifier, unique within the batch.
        messages: Conversation messages for the completion.
        max_tokens: Maximum tokens for the completion response.
        temperature: Sampling temperature.
        extra: Arbitrary provider-specific parameters.
    """

    custom_id: str
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class BatchResult:
    """Result for a single request within a completed batch.

    Args:
        custom_id: The caller-defined ID from the original request.
        status: Whether this individual request succeeded.
        content: The assistant response text, if succeeded.
        error: Error detail string, if errored.
        usage: Token usage dict from the provider, if available.
        raw: The full raw response object from the provider.
    """

    custom_id: str
    status: BatchResultStatus
    content: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None
    raw: Any | None = None


@dataclass
class BatchInfo:
    """Normalised metadata about a batch job.

    Args:
        batch_id: Provider-assigned batch identifier.
        status: Normalised lifecycle status.
        provider: Provider name string (e.g. "openai").
        created_at: Unix timestamp of batch creation.
        total: Total requests in the batch.
        completed: Count of successfully completed requests.
        failed: Count of failed requests.
        raw: The full raw batch object from the provider.
    """

    batch_id: str
    status: BatchStatus
    provider: str
    created_at: float | None = None
    total: int | None = None
    completed: int | None = None
    failed: int | None = None
    raw: Any | None = None
