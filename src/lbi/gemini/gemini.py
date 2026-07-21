"""Google Gemini batch provider.

Uses the Gemini Batch API (``client.aio.batches``) with inline
requests. Inline responses come back in submission order without a
per-item id, so this provider records the custom_id ordering per job
and matches results back by position. If a batch is retrieved that was
not created by this instance, positional ids ('0', '1', ...) are used.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import uuid

from google import genai
from google.genai import types

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
    'GeminiBatchProvider',
]

logger = logging.getLogger(__name__)

_STATE_MAP: dict[str, BatchStatus] = {
    'JOB_STATE_PENDING': BatchStatus.PENDING,
    'JOB_STATE_QUEUED': BatchStatus.PENDING,
    'JOB_STATE_RUNNING': BatchStatus.IN_PROGRESS,
    'JOB_STATE_UPDATING': BatchStatus.IN_PROGRESS,
    'JOB_STATE_PAUSED': BatchStatus.IN_PROGRESS,
    'JOB_STATE_SUCCEEDED': BatchStatus.COMPLETED,
    'JOB_STATE_PARTIALLY_SUCCEEDED': BatchStatus.COMPLETED,
    'JOB_STATE_FAILED': BatchStatus.FAILED,
    'JOB_STATE_CANCELLED': BatchStatus.CANCELLED,
    'JOB_STATE_CANCELLING': BatchStatus.CANCELLED,
    'JOB_STATE_EXPIRED': BatchStatus.EXPIRED,
}

_ROLE_MAP: dict[str, str] = {
    'user': 'user',
    'assistant': 'model',
}


def _to_timestamp(value: dt.datetime | None) -> float | None:
    return value.timestamp() if value is not None else None


def _request_to_inline(req: BatchRequest) -> types.InlinedRequest:
    """Convert a BatchRequest into a Gemini inline request.

    System messages become ``system_instruction`` on the per-request
    config; user/assistant turns become ``contents``. Note that Gemini
    uses a single model for the whole job, so req.model is ignored.
    """
    contents: list[types.Content] = []
    system_parts: list[str] = []
    for m in req.messages:
        role = m.role.value
        if role == 'system':
            system_parts.append(m.content)
        elif role in _ROLE_MAP:
            contents.append(
                types.Content(
                    role=_ROLE_MAP[role],
                    parts=[types.Part.from_text(text=m.content)],
                )
            )
        else:
            raise ValueError(f'unsupported role: {role}')

    config = types.GenerateContentConfig(
        system_instruction='\n\n'.join(system_parts) if system_parts else None,
        max_output_tokens=req.max_tokens,
        temperature=req.temperature,
        **req.extra,
    )
    return types.InlinedRequest(contents=contents, config=config)


def _usage_dict(
    response: types.GenerateContentResponse,
) -> dict[str, int] | None:
    meta = response.usage_metadata
    if meta is None:
        return None
    usage = {
        'prompt_tokens': meta.prompt_token_count,
        'completion_tokens': meta.candidates_token_count,
        'total_tokens': meta.total_token_count,
    }
    return {k: v for k, v in usage.items() if v is not None}


class GeminiBatchProvider(BaseBatchProvider):
    """Batch provider backed by the Gemini Batch API.

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY /
            GOOGLE_API_KEY per the google-genai client defaults.
    """

    provider_name: str = 'gemini'

    def __init__(self, api_key: str | None = None) -> None:
        self._client = genai.Client(api_key=api_key)
        # Maps a batch job name to the ordered custom_ids submitted, so
        # positional inline responses can be matched back.
        self._custom_ids: dict[str, list[str]] = {}

    def _parse_job(self, job: types.BatchJob) -> BatchInfo:
        if job.name is None:
            raise ProviderError('gemini', 'batch job has no name')
        custom_ids = self._custom_ids.get(job.name)
        state = job.state.name if job.state is not None else None
        return BatchInfo(
            batch_id=job.name,
            status=_STATE_MAP.get(state, BatchStatus.PENDING)
            if state is not None
            else BatchStatus.PENDING,
            provider='gemini',
            created_at=_to_timestamp(job.create_time),
            total=len(custom_ids) if custom_ids is not None else None,
            completed=None,
            failed=None,
            raw=job,
        )

    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
    ) -> BatchInfo:
        inline_requests = [_request_to_inline(r) for r in requests]
        config = (
            types.CreateBatchJobConfig(display_name=batch_filename)
            if batch_filename
            else None
        )
        try:
            job = await self._client.aio.batches.create(
                model=model,
                src=inline_requests,
                config=config,
            )
        except Exception as exc:
            raise BatchCreationError(
                f'Gemini batch creation failed: {exc}'
            ) from exc

        if job.name is None:
            raise BatchCreationError(
                'Gemini batch creation did not return a job name'
            )
        self._custom_ids[job.name] = [r.custom_id for r in requests]
        logger.info('Gemini batch created: %s', job.name)
        return self._parse_job(job)

    async def get_batch(self, batch_id: str) -> BatchInfo:
        try:
            job = await self._client.aio.batches.get(name=batch_id)
        except Exception as exc:
            raise BatchNotFoundError(
                f'Gemini batch {batch_id} not found: {exc}'
            ) from exc
        return self._parse_job(job)

    async def get_results(self, batch_id: str) -> list[BatchResult]:
        try:
            job = await self._client.aio.batches.get(name=batch_id)
        except Exception as exc:
            raise BatchNotFoundError(
                f'Gemini batch {batch_id} not found: {exc}'
            ) from exc

        state = job.state.name if job.state is not None else None
        status = (
            _STATE_MAP.get(state, BatchStatus.PENDING)
            if state is not None
            else BatchStatus.PENDING
        )
        if status != BatchStatus.COMPLETED:
            raise ResultsNotReadyError(
                f'Batch {batch_id} status is {status.value}, not completed'
            )

        inlined = job.dest.inlined_responses if job.dest is not None else None
        if inlined is None:
            raise ProviderError(
                'gemini',
                f'Batch {batch_id} has no inline responses'
                ' (file-based output is not supported by this provider)',
            )

        custom_ids = self._custom_ids.get(batch_id)
        results: list[BatchResult] = []
        for i, item in enumerate(inlined):
            if custom_ids is not None and i < len(custom_ids):
                custom_id = custom_ids[i]
            else:
                custom_id = str(i)

            if item.response is not None:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        status=BatchResultStatus.SUCCEEDED,
                        content=item.response.text,
                        usage=_usage_dict(item.response),
                        raw=item,
                    )
                )
            else:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        status=BatchResultStatus.ERRORED,
                        error=str(item.error),
                        raw=item,
                    )
                )
        return results

    async def cancel_batch(self, batch_id: str) -> BatchInfo:
        try:
            # cancel returns None; re-fetch for the updated status.
            await self._client.aio.batches.cancel(name=batch_id)
        except Exception as exc:
            raise ProviderError(
                'gemini',
                f'Failed to cancel batch {batch_id}: {exc}',
                cause=exc,
            ) from exc
        return await self.get_batch(batch_id)

    async def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        try:
            pager = await self._client.aio.batches.list(
                config=types.ListBatchJobsConfig(page_size=limit),
            )
            infos: list[BatchInfo] = []
            async for job in pager:
                infos.append(self._parse_job(job))
                if len(infos) >= limit:
                    break
            return infos
        except Exception as exc:
            raise ProviderError(
                'gemini',
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
        api_key = os.environ.get('GEMINI_API_KEY', None)
        if api_key is None:
            raise Exception('no api key provided')

    if not batch_filename:
        batch_filename = f'batch-{uuid.uuid4()}'

    provider = GeminiBatchProvider(api_key=api_key)
    batch_info = await provider.create_batch(
        batch_requests,
        model_name,
        batch_filename,
    )
    await provider.wait_for_completion(batch_info.batch_id)
    return await provider.get_results(batch_info.batch_id)
