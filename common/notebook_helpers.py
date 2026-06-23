"""Notebook setup helpers."""

from __future__ import annotations

import importlib


HELPER_MODULES = (
    "common.related_tech",
    "common.yolo_demo",
    "common.visualization",
)


def _public_names(module):
    return getattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])


def reload_tutorial_helpers(namespace=None):
    """Reload local helper modules while editing the tutorial.

    Pass ``globals()`` from a notebook cell to refresh names imported with
    ``from common.<module> import *``.
    """
    reloaded = {}
    for module_name in HELPER_MODULES:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        reloaded[module_name.rsplit(".", 1)[-1]] = module
        if namespace is not None:
            for name in _public_names(module):
                namespace[name] = getattr(module, name)
    return reloaded
