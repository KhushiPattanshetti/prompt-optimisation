"""
pipeline_logger — shared pretty-logging package for the full E2E pipeline.

Usage from any service:
    from pipeline_logger import ServiceIOLogger, get_run_context
    from pipeline_logger.system_logger import write_csv_rows, write_batch_summary  # rl_loop_svc only
"""

from .run_context import get_run_context, init_run
from .service_io_logger import ServiceIOLogger

__all__ = ["get_run_context", "init_run", "ServiceIOLogger"]
