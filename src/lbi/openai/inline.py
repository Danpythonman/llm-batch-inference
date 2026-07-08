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

import asyncio
import logging
import time
import uuid

from openai import AsyncOpenAI
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
    ProviderError,
)

__all__: list[str] = [
    'InlineOpenAIBatchProvider',
]

logger = logging.getLogger(__name__)


class InlineOpenAIBatchProvider(BaseBatchProvider):
    provider_name: str = 'openai'

    def __init__(
        self,
        api_key: str | None = None,
        max_workers: int = 8,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._max_workers = max_workers
        # Keyed by batch_id; holds (BatchInfo, results) for inline runs.
        self._inline_store: dict[str, tuple[BatchInfo, list[BatchResult]]] = {}

    async def _call_one(self, req: BatchRequest, model: str) -> BatchResult:
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
            resp = await self._client.chat.completions.create(
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

    async def _create_inline_batch(
        self,
        requests: list[BatchRequest],
        model: str,
    ) -> BatchInfo:
        """Execute all requests immediately via the regular API."""
        batch_id = f'inline-{uuid.uuid4().hex}'
        created_at = time.time()

        tasks = [self._call_one(req, model) for req in requests]
        sem = asyncio.Semaphore(self._max_workers)

        async with sem:
            results = await asyncio.gather(*tasks)

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
        del batch_filename
        return await self._create_inline_batch(requests, model)

    async def get_batch(self, batch_id: str) -> BatchInfo:
        """Retrieve current batch status.

        For inline batches, returns the stored completed BatchInfo.
        """
        if batch_id in self._inline_store:
            return self._inline_store[batch_id][0]
        raise Exception('batch not in store')

    async def get_results(self, batch_id: str) -> list[BatchResult]:
        """Download and parse results for a completed batch.

        For inline batches, returns the stored results immediately.
        """
        if batch_id in self._inline_store:
            return list(self._inline_store[batch_id][1])
        raise Exception('batch not in store')

    async def cancel_batch(self, batch_id: str) -> BatchInfo:
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
        raise Exception('batch not in store')

    async def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        """List recent batches.

        In inline mode, returns in-memory batches sorted newest-first.
        """
        infos = [v[0] for v in self._inline_store.values()]
        return sorted(
            infos,
            key=lambda x: x.created_at or 0.0,
            reverse=True,
        )[:limit]
