"""Anthropic (Claude) batch provider.

Uses the Message Batches API. Requests are submitted inline (there is
no file upload step); the whole batch carries a single
``processing_status`` and per-request outcomes are reported in
``request_counts``. Results are streamed back as JSONL once the batch
has ended.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import uuid
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types.message import Message as AnthropicMessage
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
)
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages.message_batch import MessageBatch

from lbi.base import BaseBatchProvider
from lbi.datamodels import (
    BatchInfo,
    BatchRequest,
    BatchResult,
    BatchResultStatus,
    BatchStatus,
)
from lbi.exceptions import (
    BatchCreationError,
    BatchNotFoundError,
    ProviderError,
    ResultsNotReadyError,
)

__all__: list[str] = [
    'AnthropicBatchProvider',
]

logger = logging.getLogger(__name__)

# Anthropic reports one processing_status for the entire batch. The
# 'canceling' state is still in flight, so we keep it non-terminal and
# let polling continue until the batch has 'ended'.
_STATUS_MAP: dict[str, BatchStatus] = {
    'in_progress': BatchStatus.IN_PROGRESS,
    'canceling': BatchStatus.IN_PROGRESS,
    'ended': BatchStatus.COMPLETED,
}


def _to_timestamp(value: dt.datetime) -> float:
    """Normalize Anthropic datetimes to a Unix timestamp."""
    return value.timestamp()


def _split_messages(
    req: BatchRequest,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split into (system, messages) using Anthropic's convention.

    Anthropic takes the system prompt as a separate top-level field
    rather than as a message with role 'system'.
    """
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        role = m.role.value
        if role == 'system':
            system_parts.append(m.content)
        elif role in ('user', 'assistant'):
            messages.append({'role': role, 'content': m.content})
        else:
            raise ValueError(f'unsupported role: {role}')
    system = '\n\n'.join(system_parts) if system_parts else None
    return system, messages


def _request_to_anthropic(req: BatchRequest, model: str) -> Request:
    """Convert a BatchRequest into an Anthropic batch Request."""
    system, messages = _split_messages(req)
    params: dict[str, Any] = {
        'model': req.model or model,
        'max_tokens': req.max_tokens,
        'messages': messages,
    }
    if system is not None:
        params['system'] = system
    if req.temperature is not None:
        params['temperature'] = req.temperature
    params.update(req.extra)
    return Request(
        custom_id=req.custom_id,
        params=MessageCreateParamsNonStreaming(**params),
    )


def _extract_text(message: AnthropicMessage) -> str | None:
    """Return the first text block from an Anthropic message."""
    for block in message.content:
        text_type = block.type
        if text_type == 'text':
            return getattr(block, 'text', None)
    return None


def _parse_batch(batch: MessageBatch) -> BatchInfo:
    """Map an Anthropic MessageBatch to a normalized BatchInfo."""
    counts = batch.request_counts
    total = completed = failed = None
    total = (
        counts.processing
        + counts.succeeded
        + counts.errored
        + counts.canceled
        + counts.expired
    )
    completed = counts.succeeded
    failed = counts.errored
    return BatchInfo(
        batch_id=batch.id,
        status=_STATUS_MAP.get(batch.processing_status, BatchStatus.PENDING),
        provider='anthropic',
        created_at=_to_timestamp(batch.created_at),
        total=total,
        completed=completed,
        failed=failed,
        raw=batch,
    )


class AnthropicBatchProvider(BaseBatchProvider):
    """Batch provider backed by the Anthropic Message Batches API.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY.
    """

    provider_name: str = 'anthropic'

    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
    ) -> BatchInfo:
        del batch_filename  # Anthropic batches are inline; no upload.
        try:
            batch = await self._client.messages.batches.create(
                requests=[_request_to_anthropic(r, model) for r in requests],
            )
        except Exception as exc:
            raise BatchCreationError(
                f'Anthropic batch creation failed: {exc}'
            ) from exc
        logger.info('Anthropic batch created: %s', batch.id)
        return _parse_batch(batch)

    async def get_batch(self, batch_id: str) -> BatchInfo:
        try:
            batch = await self._client.messages.batches.retrieve(batch_id)
        except Exception as exc:
            raise BatchNotFoundError(
                f'Anthropic batch {batch_id} not found: {exc}'
            ) from exc
        return _parse_batch(batch)

    async def get_results(self, batch_id: str) -> list[BatchResult]:
        info = await self.get_batch(batch_id)
        if info.status != BatchStatus.COMPLETED:
            raise ResultsNotReadyError(
                f'Batch {batch_id} status is {info.status.value},'
                ' not completed'
            )

        try:
            stream = await self._client.messages.batches.results(batch_id)
        except Exception as exc:
            raise ProviderError(
                'anthropic',
                f'Failed to stream results for {batch_id}: {exc}',
                cause=exc,
            ) from exc

        results: list[BatchResult] = []
        async for entry in stream:
            result = entry.result
            if result.type == 'succeeded':
                message = result.message
                usage = None
                usage = {
                    'input_tokens': message.usage.input_tokens,
                    'output_tokens': message.usage.output_tokens,
                }
                results.append(
                    BatchResult(
                        custom_id=entry.custom_id,
                        status=BatchResultStatus.SUCCEEDED,
                        content=_extract_text(message),
                        usage=usage,
                        raw=entry,
                    )
                )
            else:
                # errored / canceled / expired
                detail = getattr(result, 'error', None) or result.type
                results.append(
                    BatchResult(
                        custom_id=entry.custom_id,
                        status=BatchResultStatus.ERRORED,
                        error=str(detail),
                        raw=entry,
                    )
                )
        return results

    async def cancel_batch(self, batch_id: str) -> BatchInfo:
        try:
            batch = await self._client.messages.batches.cancel(batch_id)
        except Exception as exc:
            raise ProviderError(
                'anthropic',
                f'Failed to cancel batch {batch_id}: {exc}',
                cause=exc,
            ) from exc
        return _parse_batch(batch)

    async def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        try:
            infos: list[BatchInfo] = []
            async for batch in self._client.messages.batches.list(
                limit=limit,
            ):
                infos.append(_parse_batch(batch))
                if len(infos) >= limit:
                    break
            return infos
        except Exception as exc:
            raise ProviderError(
                'anthropic',
                f'Failed to list batches: {exc}',
                cause=exc,
            ) from exc


async def run_batch(
    batch_requests: list[BatchRequest],
    model_name: str,
    api_key: str | None,
    batch_filename: str | None,
) -> list[BatchResult]:
    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY', None)
        if api_key is None:
            raise Exception('no api key provided')

    if not batch_filename:
        batch_filename = f'batch-{uuid.uuid4()}.jsonl'

    provider = AnthropicBatchProvider(api_key=api_key)
    batch_info = await provider.create_batch(
        batch_requests,
        model_name,
        batch_filename,
    )
    await provider.wait_for_completion(batch_info.batch_id)
    return await provider.get_results(batch_info.batch_id)
