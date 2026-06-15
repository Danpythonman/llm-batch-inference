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

import io
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

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


def _parse_batch_object(batch: Any) -> BatchInfo:
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

    def __init__(
        self,
        api_key: str | None = None,
        client: OpenAI | None = None,
        use_inline: bool = False,
        max_workers: int = 8,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(api_key=api_key)
        self._use_inline = use_inline
        self._max_workers = max_workers
        # Keyed by batch_id; holds (BatchInfo, results) for inline runs.
        self._inline_store: dict[str, tuple[BatchInfo, list[BatchResult]]] = {}

    def _call_one(self, req: BatchRequest, model: str) -> BatchResult:
        """Execute a single request via the Chat Completions API."""
        if 'stream' in req.extra:
            raise Exception('stream parameter not allowed')
        try:
            messages: list[ChatCompletionMessageParam] = []
            for m in req.messages:
                role = m.role.value
                if role == 'user':
                    messages.append({'role': 'user', 'content': m.content})
                elif role == 'assistant':
                    messages.append(
                        {'role': 'assistant', 'content': m.content}
                    )
                elif role == 'system':
                    messages.append({'role': 'system', 'content': m.content})
                else:
                    raise ValueError(f'unsupported role: {role}')
            resp = self._client.chat.completions.create(
                model=req.model or model,
                max_tokens=req.max_tokens,
                messages=messages,
                temperature=req.temperature,
                stream=False,
                **req.extra,
            )
            content = resp.choices[0].message.content if resp.choices else None
            usage = None
            if resp.usage:
                usage = {
                    'prompt_tokens': resp.usage.prompt_tokens,
                    'completion_tokens': resp.usage.completion_tokens,
                    'total_tokens': resp.usage.total_tokens,
                }
            return BatchResult(
                custom_id=req.custom_id,
                status=BatchResultStatus.SUCCEEDED,
                content=content,
                usage=usage,
                raw=resp,
            )
        except Exception as exc:
            logger.warning(
                'Inline request %s failed: %s',
                req.custom_id,
                exc,
            )
            return BatchResult(
                custom_id=req.custom_id,
                status=BatchResultStatus.ERRORED,
                error=str(exc),
            )

    def _create_inline_batch(
        self,
        requests: list[BatchRequest],
        model: str,
    ) -> BatchInfo:
        """Execute all requests immediately via the regular API."""
        batch_id = f'inline-{uuid.uuid4().hex}'
        created_at = time.time()
        results: list[BatchResult] = [None] * len(requests)  # type: ignore

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_idx = {
                pool.submit(self._call_one, req, model): i
                for i, req in enumerate(requests)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

        completed = sum(
            1 for r in results if r.status == BatchResultStatus.SUCCEEDED
        )
        info = BatchInfo(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED,
            provider='openai',
            created_at=created_at,
            total=len(results),
            completed=completed,
            failed=len(results) - completed,
        )
        self._inline_store[batch_id] = (info, results)
        logger.info(
            'Inline batch %s complete: %d/%d succeeded',
            batch_id,
            completed,
            len(results),
        )
        return info

    def create_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
        **kwargs: object,
    ) -> BatchInfo:
        """Submit requests as a batch.

        In inline mode, executes all requests immediately via the
        regular Chat Completions API and returns a completed BatchInfo.
        Otherwise, uploads a JSONL file and creates an OpenAI batch.
        """
        if self._use_inline:
            return self._create_inline_batch(requests, model)

        lines = [
            json.dumps(_request_to_jsonl_line(r, model)) for r in requests
        ]
        payload = '\n'.join(lines).encode()
        buf = io.BytesIO(payload)
        buf.name = batch_filename

        try:
            file_obj = self._client.files.create(
                file=buf,
                purpose='batch',
            )
            batch = self._client.batches.create(
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

    def get_batch(self, batch_id: str) -> BatchInfo:
        """Retrieve current batch status.

        For inline batches, returns the stored completed BatchInfo.
        """
        if batch_id in self._inline_store:
            return self._inline_store[batch_id][0]
        try:
            batch = self._client.batches.retrieve(batch_id)
        except Exception as exc:
            raise BatchNotFoundError(
                f'OpenAI batch {batch_id} not found: {exc}'
            ) from exc
        return _parse_batch_object(batch)

    def get_results(self, batch_id: str) -> list[BatchResult]:
        """Download and parse results for a completed batch.

        For inline batches, returns the stored results immediately.
        """
        if batch_id in self._inline_store:
            return list(self._inline_store[batch_id][1])

        info = self.get_batch(batch_id)
        if info.status != BatchStatus.COMPLETED:
            raise ResultsNotReadyError(
                f'Batch {batch_id} status is {info.status.value},'
                ' not completed'
            )

        output_file_id = getattr(info.raw, 'output_file_id', None)
        if not output_file_id:
            raise ResultsNotReadyError(f'Batch {batch_id} has no output file')

        try:
            content = self._client.files.content(
                output_file_id,
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

            if status_code == 200 and not error:
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

    def cancel_batch(self, batch_id: str) -> BatchInfo:
        """Request cancellation of an OpenAI batch.

        Raises ProviderError for inline batches, which are already
        completed synchronously and cannot be cancelled.
        """
        if batch_id in self._inline_store:
            raise ProviderError(
                'openai',
                f'Inline batch {batch_id} cannot be cancelled'
                ' (already completed synchronously)',
            )
        try:
            batch = self._client.batches.cancel(batch_id)
        except Exception as exc:
            raise ProviderError(
                'openai',
                f'Failed to cancel batch {batch_id}: {exc}',
                cause=exc,
            ) from exc
        return _parse_batch_object(batch)

    def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        """List recent batches.

        In inline mode, returns in-memory batches sorted newest-first.
        """
        if self._use_inline:
            infos = [v[0] for v in self._inline_store.values()]
            return sorted(
                infos,
                key=lambda x: x.created_at or 0.0,
                reverse=True,
            )[:limit]
        try:
            page = self._client.batches.list(limit=limit)
        except Exception as exc:
            raise ProviderError(
                'openai',
                f'Failed to list batches: {exc}',
                cause=exc,
            ) from exc
        return [_parse_batch_object(b) for b in page.data]
