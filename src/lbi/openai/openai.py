"""OpenAI batch provider.

By default uses the file-based Batch API: build a JSONL payload,
upload via the Files API, create a batch, poll, and download
the output file.

Set ``use_inline=True`` to skip the Batch API entirely and issue
every request through the regular Chat Completions API concurrently
via a thread pool. Results are available immediately after
``create_batch`` returns.
"""

from __future__ import annotations

import http
import io
import json
import logging
import os
import uuid
from typing import Any

from openai import AsyncOpenAI
from openai.types import Batch

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
from lbi.openai.inline import InlineOpenAIBatchProvider

__all__: list[str] = [
    'OpenAIBatchProvider',
]

logger = logging.getLogger(__name__)

_STATUS_MAP: dict[str, BatchStatus] = {
    'validating': BatchStatus.PENDING,
    'in_progress': BatchStatus.IN_PROGRESS,
    'finalizing': BatchStatus.IN_PROGRESS,
    'completed': BatchStatus.COMPLETED,
    'failed': BatchStatus.FAILED,
    'expired': BatchStatus.EXPIRED,
    'cancelled': BatchStatus.CANCELLED,
    'cancelling': BatchStatus.CANCELLED,
}


def _request_to_jsonl_line(
    req: BatchRequest,
    model: str,
) -> dict[str, Any]:
    """Convert a BatchRequest into an OpenAI batch JSONL line."""
    body: dict[str, Any] = {
        'model': req.model or model,
        'max_tokens': req.max_tokens,
        'messages': [
            {'role': m.role.value, 'content': m.content} for m in req.messages
        ],
    }
    if req.temperature is not None:
        body['temperature'] = req.temperature
    body.update(req.extra)
    return {
        'custom_id': req.custom_id,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': body,
    }


def _parse_batch_object(batch: Batch) -> BatchInfo:
    """Map an OpenAI Batch object to a normalized BatchInfo."""
    counts = getattr(batch, 'request_counts', None)
    return BatchInfo(
        batch_id=batch.id,
        status=_STATUS_MAP.get(batch.status, BatchStatus.PENDING),
        provider='openai',
        created_at=batch.created_at,
        total=getattr(counts, 'total', None),
        completed=getattr(counts, 'completed', None),
        failed=getattr(counts, 'failed', None),
        raw=batch,
    )


class OpenAIBatchProvider(BaseBatchProvider):
    """Batch provider backed by the OpenAI API.

    By default uses the file-based Batch API: uploads a JSONL file,
    creates a batch for /v1/chat/completions, and retrieves results
    via the Files API.

    When ``use_inline=True``, skips the Batch API and issues every
    request through the regular Chat Completions API concurrently
    using a thread pool. Results are available immediately; the
    returned BatchInfo already has status COMPLETED.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env.
        client: Pre-configured OpenAI client instance.
        use_inline: Use the regular Chat Completions API instead
            of the Batch API.
        max_workers: Thread-pool size used in inline mode.
    """

    provider_name: str = 'openai'

    def __new__(
        cls,
        use_inline: bool = False,
        api_key: str | None = None,
        max_workers: int = 8,
    ) -> OpenAIBatchProvider | InlineOpenAIBatchProvider:
        if use_inline:
            return InlineOpenAIBatchProvider(
                api_key=api_key,
                max_workers=max_workers,
            )
        return super().__new__(cls)

    def __init__(
        self,
        use_inline: bool = False,
        api_key: str | None = None,
        max_workers: int = 8,
    ) -> None:
        del use_inline
        del max_workers
        self._client = AsyncOpenAI(api_key=api_key)

    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
    ) -> BatchInfo:
        """Submit requests as a batch.

        In inline mode, executes all requests immediately via the
        regular Chat Completions API and returns a completed BatchInfo.
        Otherwise, uploads a JSONL file and creates an OpenAI batch.
        """

        lines = [
            json.dumps(_request_to_jsonl_line(r, model)) for r in requests
        ]
        payload = '\n'.join(lines).encode()
        buf = io.BytesIO(payload)
        buf.name = batch_filename

        try:
            file_obj = await self._client.files.create(
                file=buf,
                purpose='batch',
            )
            batch = await self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint='/v1/chat/completions',
                completion_window='24h',
            )
        except Exception as exc:
            raise BatchCreationError(
                f'OpenAI batch creation failed: {exc}'
            ) from exc

        logger.info('OpenAI batch created: %s', batch.id)
        return _parse_batch_object(batch)

    async def get_batch(self, batch_id: str) -> BatchInfo:
        """Retrieve current batch status.

        For inline batches, returns the stored completed BatchInfo.
        """
        try:
            batch = await self._client.batches.retrieve(batch_id)
        except Exception as exc:
            raise BatchNotFoundError(
                f'OpenAI batch {batch_id} not found: {exc}'
            ) from exc
        return _parse_batch_object(batch)

    async def get_results(self, batch_id: str) -> list[BatchResult]:
        """Download and parse results for a completed batch.

        For inline batches, returns the stored results immediately.
        """
        info = await self.get_batch(batch_id)
        if info.status != BatchStatus.COMPLETED:
            raise ResultsNotReadyError(
                f'Batch {batch_id} status is {info.status.value},'
                ' not completed'
            )

        output_file_id = getattr(info.raw, 'output_file_id', None)
        if not output_file_id:
            raise ResultsNotReadyError(f'Batch {batch_id} has no output file')

        try:
            content = (
                await self._client.files.content(
                    output_file_id,
                )
            ).content
        except Exception as exc:
            raise ProviderError(
                'openai',
                f'Failed to download output file: {exc}',
                cause=exc,
            ) from exc

        results: list[BatchResult] = []
        for line in content.decode().strip().split('\n'):
            if not line:
                continue
            row = json.loads(line)
            resp = row.get('response', {})
            body = resp.get('body', {})
            status_code = resp.get('status_code', 0)
            error = row.get('error')

            if status_code == http.HTTPStatus.OK and not error:
                choices = body.get('choices', [])
                text = choices[0]['message']['content'] if choices else None
                results.append(
                    BatchResult(
                        custom_id=row['custom_id'],
                        status=BatchResultStatus.SUCCEEDED,
                        content=text,
                        usage=body.get('usage'),
                        raw=row,
                    )
                )
            else:
                results.append(
                    BatchResult(
                        custom_id=row['custom_id'],
                        status=BatchResultStatus.ERRORED,
                        error=str(error or body),
                        raw=row,
                    )
                )
        return results

    async def cancel_batch(self, batch_id: str) -> BatchInfo:
        """Request cancellation of an OpenAI batch.

        Raises ProviderError for inline batches, which are already
        completed synchronously and cannot be cancelled.
        """
        try:
            batch = await self._client.batches.cancel(batch_id)
        except Exception as exc:
            raise ProviderError(
                'openai',
                f'Failed to cancel batch {batch_id}: {exc}',
                cause=exc,
            ) from exc
        return _parse_batch_object(batch)

    async def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        """List recent batches.

        In inline mode, returns in-memory batches sorted newest-first.
        """
        try:
            page = await self._client.batches.list(limit=limit)
        except Exception as exc:
            raise ProviderError(
                'openai',
                f'Failed to list batches: {exc}',
                cause=exc,
            ) from exc
        return [_parse_batch_object(b) for b in page.data]


async def run_batch(
    batch_requests: list[BatchRequest],
    model_name: str,
    api_key: str | None,
    batch_filename: str | None,
) -> list[BatchResult]:
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', None)
        if api_key is None:
            raise Exception('no api key provided')

    if not batch_filename:
        batch_filename = f'batch-{uuid.uuid4()}.jsonl'

    openai_batch_provider = OpenAIBatchProvider(
        api_key=api_key,
    )
    batch_info = await openai_batch_provider.create_batch(
        batch_requests,
        model_name,
        batch_filename,
    )
    await openai_batch_provider.wait_for_completion(batch_info.batch_id)
    return await openai_batch_provider.get_results(batch_info.batch_id)
