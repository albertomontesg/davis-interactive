# Python2/3 Compatibility

# pathlib
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

# Testing
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch
