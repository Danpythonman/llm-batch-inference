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
from dataclasses import dataclass, field
from typing import Callable, Any

import pytest
from dotenv import load_dotenv

from lbi.datamodels import (
    BatchRequest,
    BatchResultStatus,
    BatchStatus,
    Message,
    Role,
)
from lbi.exceptions import BatchNotFoundError, ResultsNotReadyError
from lbi.openai.openai import OpenAIBatchProvider
from lbi.anthropic.anthropic import AnthropicBatchProvider
from lbi.mistral.mistral import MistralBatchProvider
from lbi.gemini.gemini import GeminiBatchProvider

load_dotenv()

pytestmark = [
    pytest.mark.integration,
]

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


@dataclass(frozen=True)
class ProviderCase:
    name: str
    env_var: str
    factory: Callable[[str], Any]
    model: str
    unknown_batch_id: str
    create_kwargs: dict = field(default_factory=dict)


CASES = [
    ProviderCase(
        name='openai',
        env_var='OPENAI_API_KEY',
        factory=lambda k: OpenAIBatchProvider(api_key=k),
        model='gpt-4o-mini',
        unknown_batch_id='batch_does_not_exist_123',
        create_kwargs={'batch_filename': 'lbi-integration-test.jsonl'},
    ),
    ProviderCase(
        name='anthropic',
        env_var='ANTHROPIC_API_KEY',
        factory=lambda k: AnthropicBatchProvider(api_key=k),
        model='claude-haiku-4-5-20251001',
        unknown_batch_id='msgbatch_does_not_exist_123',
    ),
    ProviderCase(
        name='mistral',
        env_var='MISTRAL_API_KEY',
        factory=lambda k: MistralBatchProvider(api_key=k),
        model='mistral-small-latest',
        unknown_batch_id='msgbatch_does_not_exist_123',
    ),
    ProviderCase(
        name='gemini',
        env_var='GEMINI_API_KEY',
        factory=lambda k: GeminiBatchProvider(api_key=k),
        model='gemini-3.5-flash-lite',
        unknown_batch_id='msgbatch_does_not_exist_123',
    ),
]


@pytest.fixture(params=CASES, ids=lambda c: c.name)
def provider_case(request) -> ProviderCase:
    return request.param


@pytest.fixture
def provider(provider_case: ProviderCase):
    return provider_case.factory(os.environ[provider_case.env_var])


@pytest.mark.asyncio
async def test_full_batch_lifecycle(provider, provider_case) -> None:
    """create_batch -> wait_for_completion -> get_results, end to end."""
    requests = _make_requests(3)

    info = await provider.create_batch(
        requests,
        model=provider_case.model,
        batch_filename='lbi-integration-test.jsonl',
    )
    assert info.batch_id
    assert info.status in (BatchStatus.PENDING, BatchStatus.IN_PROGRESS)
    assert info.provider == provider_case.name

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
    provider, provider_case
) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=provider_case.model,
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
    provider, provider_case
) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=provider_case.model,
        batch_filename='lbi-integration-test-early.jsonl',
    )

    # Immediately after creation the batch cannot be COMPLETED yet.
    with pytest.raises(ResultsNotReadyError):
        await provider.get_results(info.batch_id)

    await provider.cancel_batch(info.batch_id)


@pytest.mark.asyncio
async def test_cancel_batch(provider, provider_case) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=provider_case.model,
        batch_filename='lbi-integration-test-cancel.jsonl',
    )

    cancelled = await provider.cancel_batch(info.batch_id)
    assert cancelled.status in (
        BatchStatus.CANCELLED,
        BatchStatus.IN_PROGRESS,  # cancellation can be async on OpenAI's side
    )


@pytest.mark.asyncio
async def test_list_batches_includes_created_batch(
    provider, provider_case
) -> None:
    requests = _make_requests(1)
    info = await provider.create_batch(
        requests,
        model=provider_case.model,
        batch_filename='lbi-integration-test-list.jsonl',
    )


    batches = await provider.list_batches(limit=20)
    assert any(b.batch_id == info.batch_id for b in batches)

    await provider.cancel_batch(info.batch_id)


@pytest.mark.asyncio
async def test_get_batch_unknown_id_raises(
    provider, provider_case
) -> None:
    with pytest.raises(BatchNotFoundError):
        await provider.get_batch('batch_does_not_exist_123')
