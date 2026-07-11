# -*- coding: utf-8 -*-
"""
Monkey-patch for ``transformers.utils.args_doc`` to prevent crashes in
PyInstaller frozen environments.

The ``auto_docstring`` decorator calls ``get_model_name()`` which uses
``inspect.getsourcefile()``. In a frozen app the returned path may be too
short for ``path.split(os.sep)[-3]``, causing an ``IndexError`` during
the import of ``transformers.image_processing_utils_fast``.

Import this module **before** any ``transformers`` / ``whisperx`` imports.
"""

import transformers.utils.args_doc as _args_doc
import transformers.utils as _utils          # same module that image_processing_utils_fast imports from

_original_auto_docstring = _args_doc.auto_docstring


def _safe_auto_docstring(*a, **kw):
    """Wrap ``auto_docstring`` so introspection errors don't crash imports."""
    try:
        return _original_auto_docstring(*a, **kw)
    except (IndexError, AttributeError, TypeError, ValueError):
        return lambda x: x


# Patch both namespaces — ``transformers.utils`` has its own binding
# created via ``from .args_doc import auto_docstring``.
_args_doc.auto_docstring = _safe_auto_docstring
_utils.auto_docstring = _safe_auto_docstring
