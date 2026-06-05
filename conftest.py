"""
pytest configuration for ARST.

Adds the src/ directory to sys.path so that `import arst` works
in tests whether or not the package is installed in editable mode.
"""

import sys
from pathlib import Path

# Ensure src/ is importable
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
