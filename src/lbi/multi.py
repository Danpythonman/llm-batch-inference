"""Fan a single batch of requests out across multiple provider/model targets.

Each target wraps an already-constructed provider instance, so targets can
mix providers (OpenAI vs Anthropic vs Gemini vs Mistral) or reuse one
provider with different models. Submission and collection are separate
async steps so callers can submit everything and check back on it later
in the same process; failures on one target never stop the others.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, replace
from typing import Final

from lbi.base import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_POLL_TIMEOUT,
    BaseBatchProvider,
)
from lbi.datamodels import BatchInfo, BatchRequest, BatchResult, BatchStatus

__all__: list[str] = [
    'UNSET',
    'BatchSubmission',
    'BatchTarget',
    'BatchValidationResult',
    'MultiBatchResult',
    'pre_validate',
    'run_batches',
    'run_batches_and_wait',
    'wait_and_collect',
]

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel distinguishing "no override" from "override to None"."""

    def __repr__(self) -> str:
        return 'UNSET'


UNSET: Final[_Unset] = _Unset()


@dataclass
class BatchTarget:
    """One provider/model combination to run a batch of requests against.

    Args:
        provider: An already-constructed provider instance (holds its own
            API key/client).
        model: The model identifier to use with this provider.
        label: Human-readable identifier for this target. Defaults to
            "{provider.provider_name}:{model}".
        batch_filename: Name of the batch file. Defaults to a name derived
            from the label.
        temperature: If set (including to None), overrides the
            temperature on every request for this target only, letting
            a target force-clear it (e.g. for a model that rejects
            non-default temperature). Left as UNSET (default), each
            request's own temperature is used unchanged.
    """

    provider: BaseBatchProvider
    model: str
    label: str | None = None
    batch_filename: str | None = None
    temperature: float | None | _Unset = UNSET

    @property
    def resolved_label(self) -> str:
        """The effective label, falling back to provider:model."""
        return self.label or f'{self.provider.provider_name}:{self.model}'

    def apply_overrides(
        self,
        requests: list[BatchRequest],
    ) -> list[BatchRequest]:
        """Return requests with this target's temperature override applied.

        Returns the same list unchanged when there is no override. A
        temperature of None is a real override (force-clear), distinct
        from leaving it UNSET.
        """
        if self.temperature is UNSET:
            return requests
        return [replace(r, temperature=self.temperature) for r in requests]


@dataclass
class BatchSubmission:
    """The outcome of submitting a batch to a single target.

    Args:
        target: The target this submission was created for.
        info: The provider's BatchInfo, if submission succeeded.
        error: Error detail string, if submission failed.
    """

    target: BatchTarget
    info: BatchInfo | None = None
    error: str | None = None


@dataclass
class MultiBatchResult:
    """The final outcome of one target within a multi-target run.

    Args:
        label: The target's resolved label.
        provider: The provider name (e.g. "openai").
        model: The model identifier used.
        status: Final normalized batch status, or None if the batch was
            never successfully created.
        results: Per-request results, if the batch completed and results
            were retrieved.
        error: Error detail string, if submission or collection failed.
    """

    label: str
    provider: str
    model: str
    status: BatchStatus | None
    results: list[BatchResult] | None = None
    error: str | None = None


@dataclass
class BatchValidationResult:
    """The outcome of validating requests against a single target.

    Args:
        target: The target that was checked.
        error: Error detail string, if the target's provider rejected
            the requests as constructed. None if the check passed.
    """

    target: BatchTarget
    error: str | None = None

    @property
    def valid(self) -> bool:
        """Whether this target passed validation."""
        return self.error is None


def pre_validate(
    requests: list[BatchRequest],
    targets: list[BatchTarget],
) -> list[BatchValidationResult]:
    """Check requests against every target's known constraints.

    Performs no I/O — each target's provider only checks constraints
    it already knows about (e.g. a model that rejects non-default
    temperature), so this can be run before run_batches_and_wait to
    catch that class of failure without submitting anything. It is not
    a full pre-flight check: passing does not guarantee the provider
    will accept the batch.

    Args:
        requests: The requests that would be sent to every target.
        targets: The provider/model combinations to check.

    Returns:
        One BatchValidationResult per target, in the same order as
        targets.
    """
    results: list[BatchValidationResult] = []
    for target in targets:
        try:
            target.provider.validate_requests(
                target.apply_overrides(requests),
                target.model,
            )
        except Exception as exc:
            results.append(
                BatchValidationResult(target=target, error=str(exc))
            )
        else:
            results.append(BatchValidationResult(target=target))
    return results


async def _submit_one(
    target: BatchTarget,
    requests: list[BatchRequest],
) -> BatchSubmission:
    filename = (
        target.batch_filename
        or f'batch-{target.resolved_label}-{uuid.uuid4()}.jsonl'
    )
    try:
        info = await target.provider.create_batch(
            target.apply_overrides(requests),
            target.model,
            filename,
        )
    except Exception as exc:
        logger.warning(
            'Batch submission failed for %s: %s',
            target.resolved_label,
            exc,
        )
        return BatchSubmission(target=target, error=str(exc))
    return BatchSubmission(target=target, info=info)


async def run_batches(
    requests: list[BatchRequest],
    targets: list[BatchTarget],
) -> list[BatchSubmission]:
    """Submit the same batch of requests to every target concurrently.

    A failure submitting one target does not affect the others; it is
    recorded on that target's BatchSubmission instead of raising.

    Args:
        requests: The requests to send to every target.
        targets: The provider/model combinations to run against.

    Returns:
        One BatchSubmission per target, in the same order as targets.
    """
    return list(
        await asyncio.gather(*(_submit_one(t, requests) for t in targets))
    )


async def _collect_one(
    submission: BatchSubmission,
    poll_interval: float,
    timeout: float,
) -> MultiBatchResult:
    target = submission.target
    if submission.info is None:
        return MultiBatchResult(
            label=target.resolved_label,
            provider=target.provider.provider_name,
            model=target.model,
            status=None,
            error=submission.error,
        )
    try:
        final_info = await target.provider.wait_for_completion(
            submission.info.batch_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        results = await target.provider.get_results(final_info.batch_id)
    except Exception as exc:
        logger.warning(
            'Batch collection failed for %s: %s',
            target.resolved_label,
            exc,
        )
        return MultiBatchResult(
            label=target.resolved_label,
            provider=target.provider.provider_name,
            model=target.model,
            status=submission.info.status,
            error=str(exc),
        )
    return MultiBatchResult(
        label=target.resolved_label,
        provider=target.provider.provider_name,
        model=target.model,
        status=final_info.status,
        results=results,
    )


async def wait_and_collect(
    submissions: list[BatchSubmission],
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_POLL_TIMEOUT,
) -> list[MultiBatchResult]:
    """Wait for every submitted target to finish and collect its results.

    A failure or timeout on one target does not affect the others; it is
    recorded on that target's MultiBatchResult instead of raising.

    Args:
        submissions: Submissions returned by run_batches.
        poll_interval: Seconds between status checks, per target.
        timeout: Maximum seconds to wait per target.

    Returns:
        One MultiBatchResult per submission, in the same order.
    """
    return list(
        await asyncio.gather(
            *(_collect_one(s, poll_interval, timeout) for s in submissions)
        )
    )


async def run_batches_and_wait(
    requests: list[BatchRequest],
    targets: list[BatchTarget],
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_POLL_TIMEOUT,
) -> list[MultiBatchResult]:
    """Submit the same batch to every target and wait for all to finish.

    Args:
        requests: The requests to send to every target.
        targets: The provider/model combinations to run against.
        poll_interval: Seconds between status checks, per target.
        timeout: Maximum seconds to wait per target.

    Returns:
        One MultiBatchResult per target, in the same order as targets.
    """
    submissions = await run_batches(requests, targets)
    return await wait_and_collect(submissions, poll_interval, timeout)
