"""Opt-in runtime observation callbacks with no effect when uninstalled."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_OBSERVER = ContextVar("wan_runtime_evidence_observer", default=None)
_CONTEXT = ContextVar("wan_runtime_evidence_context", default={})


def runtime_observer_installed():
    return _OBSERVER.get() is not None


@contextmanager
def install_runtime_observer(observer, *, rank):
    if observer is None or not callable(observer):
        raise TypeError("runtime evidence observer must be callable")
    if runtime_observer_installed():
        raise RuntimeError("a runtime evidence observer is already installed")
    observer_token = _OBSERVER.set(observer)
    context_token = _CONTEXT.set({"rank": rank})
    try:
        emit_runtime_observation("observer-installed")
        yield observer
        emit_runtime_observation("observer-completed")
    except Exception as error:
        emit_runtime_observation(
            "observer-failed", error_type=type(error).__name__
        )
        raise
    finally:
        _CONTEXT.reset(context_token)
        _OBSERVER.reset(observer_token)


@contextmanager
def runtime_observation_scope(**coordinates):
    observer = _OBSERVER.get()
    if observer is None:
        yield
        return
    current = _CONTEXT.get()
    overlap = set(current) & set(coordinates)
    if overlap:
        raise ValueError(f"runtime evidence coordinates already set: {sorted(overlap)}")
    token = _CONTEXT.set(current | coordinates)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def emit_runtime_observation(event, **values):
    observer = _OBSERVER.get()
    if observer is None:
        return False
    if not isinstance(event, str) or not event:
        raise ValueError("runtime observation event must be a nonempty string")
    overlap = set(_CONTEXT.get()) & set(values)
    if overlap:
        raise ValueError(f"runtime observation values duplicate context: {sorted(overlap)}")
    observer({"event": event, **_CONTEXT.get(), **values})
    return True
