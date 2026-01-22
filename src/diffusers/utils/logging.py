# coding=utf-8
"""Logging utilities."""

import logging
import os
import sys
import threading
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from typing import Dict, Optional

from tqdm import auto as tqdm_lib

from .distributed_utils import is_torch_dist_rank_zero

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING

_tqdm_active = True
_rank_zero_filter = None

class _RankZeroFilter(logging.Filter):
    def filter(self, record):
        # Always allow rank-zero logs, but keep de...
        return is_torch_dist_rank_zero() or record.levelno <= logging.DEBUG

def _ensure_rank_zero_filter(logger: logging.Logger) -> None:
    global _rank_zero_filter

    if _rank_zero_filter is None:
        _rank_zero_filter = _RankZeroFilter()

    if not any(isinstance(f, _RankZeroFilter) for f in logger.filters):
        logger.addFilter(_rank_zero_filter)

def _get_default_logging_level() -> int:
    class _RankZeroFilter(logging.Filter):
    def filter(self, record):
        # Always allow rank-zero logs, but keep de...
        return is_torch_dist_rank_zero() or record.levelno <= logging.DEBUG

def _ensure_rank_zero_filter(logger: logging.Logger) -> None:
    global _rank_zero_filter

    if _rank_zero_filter is None:
        _rank_zero_filter = _RankZeroFilter()

    if not any(isinstance(f, _RankZeroFilter) for f in logger.filters):
        logger.addFilter(_rank_zero_filter)

def _get_default_logging_level() -> int:
    
    class _RankZeroFilter(logging.Filter):
    def filter(self, record):
        # Always allow rank-zero logs, but keep de...
        return is_torch_dist_rank_zero() or record.levelno <= logging.DEBUG

def _ensure_rank_zero_filter(logger: logging.Logger) -> None:
    global _rank_zero_filter

    if _rank_zero_filter is None:
        _rank_zero_filter = _RankZeroFilter()

    if not any(isinstance(f, _RankZeroFilter) for f in logger.filters):
        logger.addFilter(_rank_zero_filter)

def _get_default_logging_level() -> int:
    class _RankZeroFilter(logging.Filter):
    def filter(self, record):
        # Always allow rank-zero logs, but keep de...
        return is_torch_dist_rank_zero() or record.levelno <= logging.DEBUG

def _ensure_rank_zero_filter(logger: logging.Logger) -> None:
    global _rank_zero_filter

    if _rank_zero_filter is None:
        _rank_zero_filter = _RankZeroFilter()

    if not any(isinstance(f, _RankZeroFilter) for f in logger.filters):
        logger.addFilter(_rank_zero_filter)

def _get_default_logging_level() -> int:

    env_level_str = os.getenv("DIFFUSERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option DIFFUSERS_VERBOSITY={env_level_str}, has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level

def _get_library_name() -> str:
    return __name__.split(".")[0]

def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())

def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.

        if sys.stderr:  # only if sys.stderr exists, e.g. when not using pythonw in windows
            _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False
        _ensure_rank_zero_filter(library_root_logger)

def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None

def get_log_levels_dict() -> Dict[str, int]:
    return log_levels

def get_logger(name: Optional[str] = None) -> logging.Logger:


    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    logger = logging.getLogger(name)
    _ensure_rank_zero_filter(logger)
    return logger

def get_verbosity() -> int:
    Return the current level for the 🤗 Diffusers' root logger as an `int`.
