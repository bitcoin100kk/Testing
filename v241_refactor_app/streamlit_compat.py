from __future__ import annotations

from contextlib import contextmanager

try:  # pragma: no cover
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyStreamlit:
        session_state = _SessionState()

        def cache_data(self, *args, **kwargs):
            def decorator(func):
                func.clear = lambda: None
                return func
            return decorator

        def cache_resource(self, *args, **kwargs):
            def decorator(func):
                func.clear = lambda: None
                return func
            return decorator

        def tabs(self, labels):
            return [_DummyTab() for _ in labels]

        def columns(self, spec):
            count = spec if isinstance(spec, int) else len(spec)
            return [_DummyTab() for _ in range(count)]

        @contextmanager
        def spinner(self, *args, **kwargs):
            yield self

        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return kwargs.get("value")
            return _noop

    st = _DummyStreamlit()
