""" Logging module for the package

The logging will be handled with the abseil library.
An addition is being made in order to allow different levels of verbosity.
"""
from __future__ import absolute_import

from absl import flags
from absl.logging import (DEBUG, ERROR, FATAL, INFO, WARN, WARNING, debug,
                          error, fatal, info, set_verbosity, skip_log_prefix,
                          warn, warning)

# This removes warning and redirection to stderr
flags.FLAGS.mark_as_parsed()

# Define verbosity
_VERBOSITY_INFO_LEVEL = 0


def set_info_level(level):
    global _VERBOSITY_INFO_LEVEL
    set_verbosity(INFO)
    _VERBOSITY_INFO_LEVEL = level


def verbose(msg, level=0):
    if level <= _VERBOSITY_INFO_LEVEL:
        info(msg)


# Register frame to not show that verbose messages come from this file
skip_log_prefix(verbose)
