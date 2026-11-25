"""One deadline and whole-directory quota, including pre-spawn work."""
from pathlib import Path
import time
from .durable_attempt_budget_v2 import BudgetError


def check_scope(folder,files,limits,started):
    if time.monotonic()-started>limits.timeout_seconds:
        raise BudgetError('processing_timeout')
    try:total,count=files.output_size(Path(folder))
    except OSError as error:raise BudgetError('budget_monitor_failed') from error
    if time.monotonic()-started>limits.timeout_seconds:
        raise BudgetError('processing_timeout')
    if total>limits.max_output_bytes or count>limits.max_output_files:
        raise BudgetError('output_limit')
    return total,count
