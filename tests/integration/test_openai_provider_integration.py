"""End-to-end integration test for OpenAIBatchProvider.

Runs real batches against the OpenAI Batch API. Costs money and can
take minutes to hours to complete (OpenAI's completion_window is up
to 24h, though small batches usually finish in a few minutes).

Not part of the normal test suite. Run explicitly for releases:

    OPENAI_API_KEY=sk-... pytest -m integration \
        test_openai_batch_integration.py -v -s

Assumptions about lbi.datamodels not visible in the provider source
(Message, MessageRole) are marked below — adjust names/imports if
they differ in the actual package.
"""

from __future__ import annotations

import os

import pytest

from lbi.datamodels import (
    BatchRequest,
    BatchResultStatus,
    BatchStatus,
    Message,
    Role,
)
from lbi.exceptions import BatchNotFoundError, ResultsNotReadyError
from lbi.openai.openai import OpenAIBatchProvider

pytestmark = [
    pytest.mark.integration,
]

MODEL = 'gpt-4o-mini'
POLL_TIMEOUT_S = 60 * 30
POLL_INTERVAL_S = 10


def _make_requests(n: int) -> list[BatchRequest]:
    return [
        BatchRequest(
            custom_id=f'test-{i}',
            max_tokens=16,
            temperature=0,
            messages=[
                Message(
                    role=Role.USER,
                    content=f'Reply with exactly the digit {i}.',
                ),
            ],
        )
        for i in range(n)
    ]


@pytest.fixture
def provider() -> OpenAIBatchProvider:
    return OpenAIBatchProvider(api_key=os.environ['OPENAI_API_KEY'])


@pytest.mark.asyncio
async def test_full_batch_lifecycle(provider: OpenAIBatchProvider) -> None:
    """create_batch -> wait_for_completion -> get_results, end to end."""
    requests = _make_requests(3)

    info = await provider.create_batch(
        requests,
        model=MODEL,
        batch_filename='lbi-integration-test.jsonl',
    )
    assert info.batch_id
    assert info.status in (BatchStatus.PENDING, BatchStatus.IN_PROGRESS)
    assert info.provider == 'openai'

    final_info = await provider.wait_for_completion(
        info.batch_id,
        timeout=POLL_TIMEOUT_S,
        poll_interval=POLL_INTERVAL_S,
    )
    assert final_info.status == BatchStatus.COMPLETED

    results = await provider.get_results(info.batch_id)
    assert len(results) == len(requests)

    by_id = {r.custom_id: r for r in results}
    assert set(by_id) == {r.custom_id for r in requests}

    for r in results:
        assert r.status == BatchResultStatus.SUCCEEDED
        assert r.content
        assert r.usage is not None


@pytest.mark.asyncio
async def test_get_batch_status_reflects_real_batch(
    provider: OpenAIBatchProvider,
) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=MODEL,
        batch_filename='lbi-integration-test-status.jsonl',
    )

    fetched = await provider.get_batch(info.batch_id)
    assert fetched.batch_id == info.batch_id
    assert fetched.status in (
        BatchStatus.PENDING,
        BatchStatus.IN_PROGRESS,
        BatchStatus.COMPLETED,
    )


@pytest.mark.asyncio
async def test_get_results_before_completion_raises(
    provider: OpenAIBatchProvider,
) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=MODEL,
        batch_filename='lbi-integration-test-early.jsonl',
    )

    # Immediately after creation the batch cannot be COMPLETED yet.
    with pytest.raises(ResultsNotReadyError):
        await provider.get_results(info.batch_id)

    await provider.cancel_batch(info.batch_id)


@pytest.mark.asyncio
async def test_cancel_batch(provider: OpenAIBatchProvider) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=MODEL,
        batch_filename='lbi-integration-test-cancel.jsonl',
    )

    cancelled = await provider.cancel_batch(info.batch_id)
    assert cancelled.status in (
        BatchStatus.CANCELLED,
        BatchStatus.IN_PROGRESS,  # cancellation can be async on OpenAI's side
    )


@pytest.mark.asyncio
async def test_list_batches_includes_created_batch(
    provider: OpenAIBatchProvider,
) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=MODEL,
        batch_filename='lbi-integration-test-list.jsonl',
    )

    batches = await provider.list_batches(limit=20)
    assert any(b.batch_id == info.batch_id for b in batches)

    await provider.cancel_batch(info.batch_id)


@pytest.mark.asyncio
async def test_get_batch_unknown_id_raises(
    provider: OpenAIBatchProvider,
) -> None:
    with pytest.raises(BatchNotFoundError):
        await provider.get_batch('batch_does_not_exist_123')
