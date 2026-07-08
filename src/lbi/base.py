"""Abstract base class for batch LLM providers."""

from __future__ import annotations

import abc
import asyncio
import logging
import time

from lbi.datamodels import (
    BatchInfo,
    BatchRequest,
    BatchResult,
    BatchStatus,
)
from lbi.exceptions import BatchPollTimeoutError

__all__: list[str] = [
    'BaseBatchProvider',
]


DEFAULT_POLL_INTERVAL: float = 10.0
DEFAULT_POLL_TIMEOUT: float = 86400.0  # 24 hours

logger = logging.getLogger(__name__)


class BaseBatchProvider(abc.ABC):
    """Interface that every batch provider must implement.

    Subclasses wrap a specific provider SDK and translate between
    the normalized LBI models and the provider's native API.
    """

    # Subclasses should set this to a short lowercase name.
    provider_name: str = 'base'

    @abc.abstractmethod
    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
    ) -> BatchInfo:
        """Submit a list of requests as a new batch job.

        Args:
            requests: The requests to include in the batch.
            model: The model identifier to use.
            batch_filename: The name of the batch file.

        Returns:
            A BatchInfo with the provider-assigned batch ID
            and initial status.

        Raises:
            BatchCreationError: If the provider rejects the batch.
        """

    @abc.abstractmethod
    async def get_batch(self, batch_id: str) -> BatchInfo:
        """Retrieve current metadata for an existing batch.

        Args:
            batch_id: The provider-assigned batch identifier.

        Returns:
            Updated BatchInfo reflecting current status.

        Raises:
            BatchNotFoundError: If the batch ID is unknown.
        """

    @abc.abstractmethod
    async def get_results(self, batch_id: str) -> list[BatchResult]:
        """Download results for a completed batch.

        Args:
            batch_id: The provider-assigned batch identifier.

        Returns:
            A list of BatchResult, one per original request.

        Raises:
            ResultsNotReadyError: If the batch has not finished.
        """

    @abc.abstractmethod
    async def cancel_batch(self, batch_id: str) -> BatchInfo:
        """Request cancellation of a running batch.

        Args:
            batch_id: The provider-assigned batch identifier.

        Returns:
            Updated BatchInfo (status may be cancelling).
        """

    @abc.abstractmethod
    async def list_batches(
        self,
        limit: int = 20,
    ) -> list[BatchInfo]:
        """List recent batch jobs.

        Args:
            limit: Maximum number of batches to return.

        Returns:
            A list of BatchInfo ordered newest-first.
        """

    # --- Concrete helpers ---
    async def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> BatchInfo:
        """Block until the batch reaches a terminal state.

        Polls get_batch at the given interval until the status
        is no longer pending or in_progress, or until timeout.

        Args:
            batch_id: The provider-assigned batch identifier.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising.

        Returns:
            The final BatchInfo.

        Raises:
            BatchPollTimeout: If timeout is exceeded.
        """
        _terminal = {
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED,
        }
        start = time.monotonic()
        while True:
            info = await self.get_batch(batch_id)
            if info.status in _terminal:
                logger.info(
                    'Batch %s reached terminal status: %s',
                    batch_id,
                    info.status.value,
                )
                return info

            elapsed = time.monotonic() - start
            if elapsed + poll_interval > timeout:
                raise BatchPollTimeoutError(
                    f'Batch {batch_id} did not complete within'
                    f' {timeout}s (last status: {info.status.value})'
                )

            logger.debug(
                'Batch %s status=%s, polling again in %.1fs',
                batch_id,
                info.status.value,
                poll_interval,
            )
            await asyncio.sleep(poll_interval)

    async def submit_and_wait(
        self,
        requests: list[BatchRequest],
        model: str,
        batch_filename: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> list[BatchResult]:
        """Convenience: create a batch, wait, and return results.

        Args:
            requests: The requests to batch.
            model: The model identifier.
            batch_filename: The name of the batch file.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.
            **kwargs: Forwarded to create_batch.

        Returns:
            A list of BatchResult for the completed batch.

        Raises:
            BatchCreationError: On submission failure.
            BatchPollTimeout: If the batch does not finish.
            ResultsNotReadyError: Should not occur normally.
        """
        info = await self.create_batch(requests, model, batch_filename)
        logger.info(
            'Created batch %s on %s, waiting for completion',
            info.batch_id,
            self.provider_name,
        )
        await self.wait_for_completion(
            info.batch_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        return await self.get_results(info.batch_id)
