from __future__ import absolute_import

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
    from pathlib2 import Path

__version__ = '0.0.2dev5'
