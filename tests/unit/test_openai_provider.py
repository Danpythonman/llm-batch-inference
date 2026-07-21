"""Unit tests for OpenAIBatchProvider (lbi.openai.batch).

Mocks the OpenAI SDK client entirely - no network calls, no cost, no
`integration` marker needed. For real end-to-end coverage see
tests/integration/test_openai_provider_integration.py.

Assumptions about lbi.datamodels not visible in the provider source
(Message, MessageRole) are marked below - adjust if names differ.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import Batch

from lbi.datamodels import (
    BatchRequest,
    BatchResultStatus,
    BatchStatus,
    Message,  # ASSUMPTION: model with role, content
    Role,  # ASSUMPTION: enum with .USER member
)
from lbi.exceptions import (
    BatchCreationError,
    BatchNotFoundError,
    ProviderError,
    ResultsNotReadyError,
)
from lbi.openai.inline import InlineOpenAIBatchProvider
from lbi.openai.openai import (
    OpenAIBatchProvider,
    _parse_batch_object,
    _request_to_jsonl_line,
    run_batch,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_batch(
    *,
    id: str = 'batch_123',
    status: str = 'completed',
    created_at: int = 1_700_000_000,
    output_file_id: str | None = 'file_out_1',
    total: int = 3,
    completed: int = 3,
    failed: int = 0,
) -> Batch:
    """Minimal stand-in for openai.types.Batch."""
    counts = SimpleNamespace(total=total, completed=completed, failed=failed)
    return cast(
        Batch,
        SimpleNamespace(
            id=id,
            status=status,
            created_at=created_at,
            output_file_id=output_file_id,
            request_counts=counts,
        ),
    )


def _make_requests(n: int = 2) -> list[BatchRequest]:
    return [
        BatchRequest(
            custom_id=f'req-{i}',
            max_tokens=16,
            temperature=0,
            messages=[Message(role=Role.USER, content=f'hello {i}')],
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_client(mocker):
    """Patch AsyncOpenAI so OpenAIBatchProvider talks to a mock client."""
    client = MagicMock()
    client.files.create = AsyncMock()
    client.files.content = AsyncMock()
    client.batches.create = AsyncMock()
    client.batches.retrieve = AsyncMock()
    client.batches.cancel = AsyncMock()
    client.batches.list = AsyncMock()
    mocker.patch('lbi.openai.openai.AsyncOpenAI', return_value=client)
    return client


@pytest.fixture
def provider(mock_client) -> OpenAIBatchProvider:
    return OpenAIBatchProvider(api_key='sk-test')


# ---------------------------------------------------------------------------
# __new__ dispatch
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_returns_openai_batch_provider(self, mock_client):
        p = OpenAIBatchProvider(api_key='sk-test')
        assert isinstance(p, OpenAIBatchProvider)
        assert not isinstance(p, InlineOpenAIBatchProvider)

    def test_use_inline_returns_inline_provider(self, mocker):
        mocker.patch(
            'lbi.openai.inline.InlineOpenAIBatchProvider.__init__',
            return_value=None,
        )
        p = OpenAIBatchProvider(
            use_inline=True, api_key='sk-test', max_workers=4
        )
        assert isinstance(p, InlineOpenAIBatchProvider)


# ---------------------------------------------------------------------------
# _request_to_jsonl_line
# ---------------------------------------------------------------------------


class TestRequestToJsonlLine:
    def test_basic_fields(self):
        req = BatchRequest(
            custom_id='abc',
            max_tokens=10,
            messages=[Message(role=Role.USER, content='hi')],
        )
        line = _request_to_jsonl_line(req, model='gpt-4o-mini')

        assert line['custom_id'] == 'abc'
        assert line['method'] == 'POST'
        assert line['url'] == '/v1/chat/completions'
        assert line['body']['model'] == 'gpt-4o-mini'
        assert line['body']['max_tokens'] == 10
        assert line['body']['messages'] == [{'role': 'user', 'content': 'hi'}]
        assert 'temperature' not in line['body']

    def test_per_request_model_overrides_default(self):
        req = BatchRequest(
            custom_id='abc',
            model='gpt-4o',
            max_tokens=10,
            messages=[Message(role=Role.USER, content='hi')],
        )
        line = _request_to_jsonl_line(req, model='gpt-4o-mini')
        assert line['body']['model'] == 'gpt-4o'

    def test_temperature_included_when_set(self):
        req = BatchRequest(
            custom_id='abc',
            max_tokens=10,
            temperature=0.7,
            messages=[Message(role=Role.USER, content='hi')],
        )
        line = _request_to_jsonl_line(req, model='gpt-4o-mini')
        assert line['body']['temperature'] == 0.7

    def test_extra_fields_merged_into_body(self):
        req = BatchRequest(
            custom_id='abc',
            max_tokens=10,
            messages=[Message(role=Role.USER, content='hi')],
            extra={'top_p': 0.9, 'seed': 42},
        )
        line = _request_to_jsonl_line(req, model='gpt-4o-mini')
        assert line['body']['top_p'] == 0.9
        assert line['body']['seed'] == 42


# ---------------------------------------------------------------------------
# _parse_batch_object
# ---------------------------------------------------------------------------


class TestParseBatchObject:
    @pytest.mark.parametrize(
        'raw_status,expected',
        [
            ('validating', BatchStatus.PENDING),
            ('in_progress', BatchStatus.IN_PROGRESS),
            ('finalizing', BatchStatus.IN_PROGRESS),
            ('completed', BatchStatus.COMPLETED),
            ('failed', BatchStatus.FAILED),
            ('expired', BatchStatus.EXPIRED),
            ('cancelled', BatchStatus.CANCELLED),
            ('cancelling', BatchStatus.CANCELLED),
        ],
    )
    def test_status_mapping(self, raw_status, expected):
        batch = _fake_batch(status=raw_status)
        info = _parse_batch_object(batch)
        assert info.status == expected

    def test_unknown_status_defaults_to_pending(self):
        batch = _fake_batch(status='some_new_status_openai_added')
        info = _parse_batch_object(batch)
        assert info.status == BatchStatus.PENDING

    def test_fields_mapped(self):
        batch = _fake_batch(id='batch_xyz', total=5, completed=2, failed=1)
        info = _parse_batch_object(batch)
        assert info.batch_id == 'batch_xyz'
        assert info.provider == 'openai'
        assert info.total == 5
        assert info.completed == 2
        assert info.failed == 1
        assert info.raw is batch

    def test_missing_request_counts_yields_none(self):
        batch = cast(
            Batch,
            SimpleNamespace(id='batch_1', status='completed', created_at=0),
        )
        info = _parse_batch_object(batch)
        assert info.total is None
        assert info.completed is None
        assert info.failed is None


# ---------------------------------------------------------------------------
# create_batch
# ---------------------------------------------------------------------------


class TestCreateBatch:
    async def test_uploads_file_then_creates_batch(
        self, provider, mock_client
    ):
        mock_client.files.create.return_value = SimpleNamespace(id='file_1')
        mock_client.batches.create.return_value = _fake_batch(
            status='validating'
        )

        info = await provider.create_batch(
            _make_requests(2),
            model='gpt-4o-mini',
            batch_filename='b.jsonl',
        )

        assert info.status == BatchStatus.PENDING
        mock_client.files.create.assert_awaited_once()
        kwargs = mock_client.files.create.call_args.kwargs
        assert kwargs['purpose'] == 'batch'
        assert kwargs['file'].name == 'b.jsonl'

        mock_client.batches.create.assert_awaited_once_with(
            input_file_id='file_1',
            endpoint='/v1/chat/completions',
            completion_window='24h',
        )

    async def test_payload_has_one_jsonl_line_per_request(
        self,
        provider,
        mock_client,
    ):
        mock_client.files.create.return_value = SimpleNamespace(id='file_1')
        mock_client.batches.create.return_value = _fake_batch()

        requests = _make_requests(3)
        await provider.create_batch(
            requests,
            model='gpt-4o-mini',
            batch_filename='b.jsonl',
        )

        uploaded = mock_client.files.create.call_args.kwargs['file']
        lines = uploaded.getvalue().decode().strip().split('\n')
        assert len(lines) == 3
        custom_ids = {json.loads(l)['custom_id'] for l in lines}
        assert custom_ids == {r.custom_id for r in requests}

    async def test_upload_failure_raises_batch_creation_error(
        self,
        provider,
        mock_client,
    ):
        mock_client.files.create.side_effect = RuntimeError('boom')

        with pytest.raises(BatchCreationError):
            await provider.create_batch(
                _make_requests(1),
                model='gpt-4o-mini',
                batch_filename='b.jsonl',
            )

    async def test_batch_creation_failure_raises_batch_creation_error(
        self,
        provider,
        mock_client,
    ):
        mock_client.files.create.return_value = SimpleNamespace(id='file_1')
        mock_client.batches.create.side_effect = RuntimeError('boom')

        with pytest.raises(BatchCreationError):
            await provider.create_batch(
                _make_requests(1),
                model='gpt-4o-mini',
                batch_filename='b.jsonl',
            )


# ---------------------------------------------------------------------------
# get_batch
# ---------------------------------------------------------------------------


class TestGetBatch:
    async def test_returns_parsed_info(self, provider, mock_client):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='in_progress'
        )
        info = await provider.get_batch('batch_123')
        assert info.status == BatchStatus.IN_PROGRESS
        mock_client.batches.retrieve.assert_awaited_once_with('batch_123')

    async def test_not_found_raises(self, provider, mock_client):
        mock_client.batches.retrieve.side_effect = RuntimeError('404')
        with pytest.raises(BatchNotFoundError):
            await provider.get_batch('does_not_exist')


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------


class TestGetResults:
    async def test_raises_when_not_completed(self, provider, mock_client):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='in_progress'
        )
        with pytest.raises(ResultsNotReadyError):
            await provider.get_results('batch_123')

    async def test_raises_when_no_output_file(self, provider, mock_client):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='completed',
            output_file_id=None,
        )
        with pytest.raises(ResultsNotReadyError):
            await provider.get_results('batch_123')

    async def test_download_failure_raises_provider_error(
        self,
        provider,
        mock_client,
    ):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='completed'
        )
        mock_client.files.content.side_effect = RuntimeError('boom')
        with pytest.raises(ProviderError):
            await provider.get_results('batch_123')

    async def test_parses_successful_results(self, provider, mock_client):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='completed'
        )
        rows = [
            {
                'custom_id': 'req-0',
                'response': {
                    'status_code': 200,
                    'body': {
                        'choices': [{'message': {'content': 'hi there'}}],
                        'usage': {'total_tokens': 5},
                    },
                },
                'error': None,
            },
        ]
        content = '\n'.join(json.dumps(r) for r in rows).encode()
        mock_client.files.content.return_value = SimpleNamespace(
            content=content
        )

        results = await provider.get_results('batch_123')

        assert len(results) == 1
        r = results[0]
        assert r.custom_id == 'req-0'
        assert r.status == BatchResultStatus.SUCCEEDED
        assert r.content == 'hi there'
        assert r.usage == {'total_tokens': 5}

    async def test_parses_errored_results(self, provider, mock_client):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='completed'
        )
        rows = [
            {
                'custom_id': 'req-0',
                'response': {'status_code': 400, 'body': {}},
                'error': {'message': 'bad request'},
            },
        ]
        content = '\n'.join(json.dumps(r) for r in rows).encode()
        mock_client.files.content.return_value = SimpleNamespace(
            content=content
        )

        results = await provider.get_results('batch_123')

        assert len(results) == 1
        r = results[0]
        assert r.status == BatchResultStatus.ERRORED
        assert 'bad request' in r.error

    async def test_skips_blank_lines(self, provider, mock_client):
        mock_client.batches.retrieve.return_value = _fake_batch(
            status='completed'
        )
        row = {
            'custom_id': 'req-0',
            'response': {
                'status_code': 200,
                'body': {'choices': [{'message': {'content': 'ok'}}]},
            },
            'error': None,
        }
        content = ('\n' + json.dumps(row) + '\n\n').encode()
        mock_client.files.content.return_value = SimpleNamespace(
            content=content
        )

        results = await provider.get_results('batch_123')
        assert len(results) == 1


# ---------------------------------------------------------------------------
# cancel_batch
# ---------------------------------------------------------------------------


class TestCancelBatch:
    async def test_returns_parsed_info(self, provider, mock_client):
        mock_client.batches.cancel.return_value = _fake_batch(
            status='cancelling'
        )
        info = await provider.cancel_batch('batch_123')
        assert info.status == BatchStatus.CANCELLED
        mock_client.batches.cancel.assert_awaited_once_with('batch_123')

    async def test_failure_raises_provider_error(self, provider, mock_client):
        mock_client.batches.cancel.side_effect = RuntimeError('boom')
        with pytest.raises(ProviderError):
            await provider.cancel_batch('batch_123')


# ---------------------------------------------------------------------------
# list_batches
# ---------------------------------------------------------------------------


class TestListBatches:
    async def test_returns_parsed_list(self, provider, mock_client):
        mock_client.batches.list.return_value = SimpleNamespace(
            data=[_fake_batch(id='b1'), _fake_batch(id='b2')],
        )
        batches = await provider.list_batches(limit=5)
        assert [b.batch_id for b in batches] == ['b1', 'b2']
        mock_client.batches.list.assert_awaited_once_with(limit=5)

    async def test_failure_raises_provider_error(self, provider, mock_client):
        mock_client.batches.list.side_effect = RuntimeError('boom')
        with pytest.raises(ProviderError):
            await provider.list_batches()


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


class TestRunBatch:
    async def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        with pytest.raises(Exception):
            await run_batch(
                _make_requests(1),
                model_name='gpt-4o-mini',
                api_key=None,
                batch_filename=None,
            )

    async def test_full_flow_uses_provided_api_key(
        self,
        mock_client,
        mocker,
    ):
        mock_client.files.create.return_value = SimpleNamespace(id='file_1')
        mock_client.batches.create.return_value = _fake_batch(
            status='validating'
        )

        mocker.patch.object(
            OpenAIBatchProvider,
            'wait_for_completion',
            AsyncMock(
                return_value=_parse_batch_object(
                    _fake_batch(status='completed')
                )
            ),
        )
        mocker.patch.object(
            OpenAIBatchProvider,
            'get_results',
            AsyncMock(return_value=['sentinel']),
        )

        results = await run_batch(
            _make_requests(1),
            model_name='gpt-4o-mini',
            api_key='sk-test',
            batch_filename=None,
        )
        assert results == ['sentinel']

    async def test_falls_back_to_env_api_key(
        self,
        mock_client,
        mocker,
        monkeypatch,
    ):
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-env')
        mock_client.files.create.return_value = SimpleNamespace(id='file_1')
        mock_client.batches.create.return_value = _fake_batch(
            status='validating'
        )
        mocker.patch.object(
            OpenAIBatchProvider, 'wait_for_completion', AsyncMock()
        )
        mocker.patch.object(
            OpenAIBatchProvider,
            'get_results',
            AsyncMock(return_value=[]),
        )

        await run_batch(
            _make_requests(1),
            model_name='gpt-4o-mini',
            api_key=None,
            batch_filename=None,
        )
        # No assertion error means the env var was picked up successfully.

    async def test_generates_default_filename_when_none_given(
        self,
        mock_client,
        mocker,
    ):
        mock_client.files.create.return_value = SimpleNamespace(id='file_1')
        mock_client.batches.create.return_value = _fake_batch(
            status='validating'
        )
        mocker.patch.object(
            OpenAIBatchProvider, 'wait_for_completion', AsyncMock()
        )
        mocker.patch.object(
            OpenAIBatchProvider,
            'get_results',
            AsyncMock(return_value=[]),
        )

        await run_batch(
            _make_requests(1),
            model_name='gpt-4o-mini',
            api_key='sk-test',
            batch_filename=None,
        )

        filename = mock_client.files.create.call_args.kwargs['file'].name
        assert filename.startswith('batch-') and filename.endswith('.jsonl')
