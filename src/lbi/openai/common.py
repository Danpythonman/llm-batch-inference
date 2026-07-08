from __future__ import annotations

import logging

from lbi.datamodels import (
    BatchStatus,
)

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
