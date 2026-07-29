"""Test-collection bootstrap for tools/memory/tests.

The monorepo's root ``pytest.ini`` sets ``import_mode = importlib`` and is
invoked from the repo root. That mode does not walk the ``__init__.py``
chain to add the repo root to ``sys.path`` the way the legacy "prepend" mode
does, and ``tools/`` itself has no ``__init__.py`` (it is an implicit
namespace package). Without the repo root on ``sys.path``, every test module
here that does ``from tools.memory... import ...`` at module scope (or
inside a test body, e.g. test_index_graceful_shutdown.py) fails with
``ModuleNotFoundError: No module named 'tools'``.

conftest.py is imported by pytest before it collects sibling test modules in
this directory, so inserting the path here — once — fixes collection and
execution for the whole package instead of requiring every test file to
carry its own sys.path boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
