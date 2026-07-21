"""Mistral batch provider.

Uploads a JSONL batch file via the Files API, creates a batch job
against ``/v1/chat/completions``, polls the job's status, then
downloads and parses the output file. The Mistral output format mirrors
OpenAI's: one JSON object per line with ``custom_id`` and a nested
``response``.
"""

from __future__ import annotations

import http
import json
import logging
import os
import uuid

from mistralai.client import Mistral
from mistralai.client.models import BatchJob

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
    'MistralBatchProvider',
]

logger = logging.getLogger(__name__)

type JSONValue = (
    str | int | float | bool | None | dict[str, JSONValue] | list[JSONValue]
)

_STATUS_MAP: dict[str, BatchStatus] = {
    'QUEUED': BatchStatus.PENDING,
    'RUNNING': BatchStatus.IN_PROGRESS,
    'CANCELLATION_REQUESTED': BatchStatus.IN_PROGRESS,
    'SUCCESS': BatchStatus.COMPLETED,
    'FAILED': BatchStatus.FAILED,
    'TIMEOUT_EXCEEDED': BatchStatus.EXPIRED,
    'CANCELLED': BatchStatus.CANCELLED,
}


def _request_to_jsonl_line(
    req: BatchRequest,
    model: str,
) -> dict[str, JSONValue]:
    """Convert a BatchRequest into a Mistral batch JSONL line."""
    body: dict[str, JSONValue] = {
        'model': req.model or model,
        'max_tokens': req.max_tokens,
        'messages': [
            {'role': m.role.value, 'content': m.content} for m in req.messages
        ],
    }
    if req.temperature is not None:
        body['temperature'] = req.temperature
    body.update(req.extra)
    return {'custom_id': req.custom_id, 'body': body}


def _parse_job(job: BatchJob) -> BatchInfo:
    """Map a Mistral BatchJob to a normalized BatchInfo."""
    return BatchInfo(
        batch_id=job.id,
        status=_STATUS_MAP.get(job.status, BatchStatus.PENDING),
        provider='mistral',
        created_at=getattr(job, 'created_at', None),
        total=getattr(job, 'total_requests', None),
        completed=getattr(job, 'succeeded_requests', None),
        failed=getattr(job, 'failed_requests', None),
        raw=job,
    )


class MistralBatchProvider(BaseBatchProvider):
    """Batch provider backed by the Mistral Batch API.

    Args:
        api_key: Mistral API key. Falls back to MISTRAL_API_KEY.
    """

    provider_name: str = 'mistral'

    def __init__(self, api_key: str | None = None) -> None:
        self._client = Mistral(
            api_key=api_key or os.environ.get('MISTRAL_API_KEY'),
        )

    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
    ) -> BatchInfo:
        lines = [
            json.dumps(_request_to_jsonl_line(r, model)) for r in requests
        ]
        payload = '\n'.join(lines).encode()

        try:
            uploaded = await self._client.files.upload_async(
                file={'file_name': batch_filename, 'content': payload},
                purpose='batch',
            )
            job = await self._client.batch.jobs.create_async(
                input_files=[uploaded.id],
                model=model,
                endpoint='/v1/chat/completions',
            )
        except Exception as exc:
            raise BatchCreationError(
                f'Mistral batch creation failed: {exc}'
            ) from exc

        logger.info('Mistral batch created: %s', job.id)
        return _parse_job(job)

    async def get_batch(self, batch_id: str) -> BatchInfo:
        try:
            job = await self._client.batch.jobs.get_async(job_id=batch_id)
        except Exception as exc:
            raise BatchNotFoundError(
                f'Mistral batch {batch_id} not found: {exc}'
            ) from exc
        return _parse_job(job)

    async def get_results(self, batch_id: str) -> list[BatchResult]:
        info = await self.get_batch(batch_id)
        if info.status != BatchStatus.COMPLETED:
            raise ResultsNotReadyError(
                f'Batch {batch_id} status is {info.status.value},'
                ' not completed'
            )

        output_file = getattr(info.raw, 'output_file', None)
        if not output_file:
            raise ResultsNotReadyError(f'Batch {batch_id} has no output file')

        try:
            resp = await self._client.files.download_async(file_id=output_file)
            content = await resp.aread()
        except Exception as exc:
            raise ProviderError(
                'mistral',
                f'Failed to download output file: {exc}',
                cause=exc,
            ) from exc

        results: list[BatchResult] = []
        for line in content.decode().strip().split('\n'):
            if not line:
                continue
            row = json.loads(line)
            resp_obj = row.get('response', {})
            body = resp_obj.get('body', {})
            status_code = resp_obj.get('status_code', 0)
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
        try:
            job = await self._client.batch.jobs.cancel_async(job_id=batch_id)
        except Exception as exc:
            raise ProviderError(
                'mistral',
                f'Failed to cancel batch {batch_id}: {exc}',
                cause=exc,
            ) from exc
        return _parse_job(job)

    async def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        try:
            resp = await self._client.batch.jobs.list_async(page_size=limit)
        except Exception as exc:
            raise ProviderError(
                'mistral',
                f'Failed to list batches: {exc}',
                cause=exc,
            ) from exc
        jobs = resp.data or []
        return [_parse_job(j) for j in jobs[:limit]]


async def run_batch(
    batch_requests: list[BatchRequest],
    model_name: str,
    api_key: str | None,
    batch_filename: str | None,
) -> list[BatchResult]:
    if not api_key:
        api_key = os.environ.get('MISTRAL_API_KEY', None)
        if api_key is None:
            raise Exception('no api key provided')

    if not batch_filename:
        batch_filename = f'batch-{uuid.uuid4()}.jsonl'

    provider = MistralBatchProvider(api_key=api_key)
    batch_info = await provider.create_batch(
        batch_requests,
        model_name,
        batch_filename,
    )
    await provider.wait_for_completion(batch_info.batch_id)
    return await provider.get_results(batch_info.batch_id)
