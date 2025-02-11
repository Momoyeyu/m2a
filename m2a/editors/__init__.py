"""Editor registry.

Model editors are imported lazily: the CRNN attack only needs the
``sed-crnn`` dependencies and the ATST-SED attack only needs the
``ATST-SED`` environment, mirroring the original per-model scripts.
"""

import importlib

_EDITORS = {
    "crnn": ("m2a.editors.crnn", "CrnnEditor"),
    "atst_sed": ("m2a.editors.atst_sed", "AtstSedEditor"),
}


def get_editor_class(model):
    if model not in _EDITORS:
        raise KeyError(f"Unknown model '{model}'. Available: {sorted(_EDITORS)}")
    module_name, class_name = _EDITORS[model]
    return getattr(importlib.import_module(module_name), class_name)
