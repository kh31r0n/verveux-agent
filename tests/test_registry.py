"""Tests for the code-name graph registry.

Covers: lazy compilation, double-checked locking under concurrent load,
the UnknownCodeNameError path, and the legacy agent_type fallback shim.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from src.graphs import registry
from src.graphs.registry import (
    CODE_NAME_REGISTRY,
    UnknownCodeNameError,
    get_or_compile_graph,
    known_code_names,
    resolve_legacy_agent_type,
    set_checkpointer,
    warm_up,
)


@pytest.fixture(autouse=True)
def _reset_registry_state():
    """Clear the lazy compile cache between tests."""
    registry._compiled_graphs.clear()
    registry._checkpointer = None
    yield
    registry._compiled_graphs.clear()
    registry._checkpointer = None


class TestRegistryShape:
    def test_seeded_code_names(self):
        assert known_code_names() == {"helena", "sofia", "giulia", "marco"}

    def test_legacy_fallback_maps_each_agent_type(self):
        assert resolve_legacy_agent_type("sales") == "helena"
        assert resolve_legacy_agent_type("school") == "sofia"
        assert resolve_legacy_agent_type("restaurant") == "giulia"
        assert resolve_legacy_agent_type("appointments") == "marco"

    def test_legacy_fallback_returns_none_for_unknown(self):
        assert resolve_legacy_agent_type("ghost") is None
        assert resolve_legacy_agent_type("") is None


class TestLazyCompilation:
    @pytest.mark.asyncio
    async def test_unknown_code_name_raises(self):
        set_checkpointer(MagicMock())
        with pytest.raises(UnknownCodeNameError) as excinfo:
            await get_or_compile_graph("luna")
        assert "luna" in str(excinfo.value)
        assert "helena" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_compiles_once_then_caches(self, monkeypatch):
        call_count = {"n": 0}
        sentinel = object()

        def fake_builder(_ckpt):
            call_count["n"] += 1
            return sentinel

        monkeypatch.setitem(CODE_NAME_REGISTRY, "helena", fake_builder)
        set_checkpointer(MagicMock())

        first = await get_or_compile_graph("helena")
        second = await get_or_compile_graph("helena")

        assert first is sentinel
        assert second is sentinel
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_double_checked_lock_under_concurrency(self, monkeypatch):
        """Concurrent get_or_compile_graph calls must invoke the builder once."""
        call_count = {"n": 0}
        sentinel = object()
        ready = asyncio.Event()

        def slow_builder(_ckpt):
            # Synchronous build but long-running enough that concurrent callers
            # all hit the locked section.
            call_count["n"] += 1
            ready.set()
            return sentinel

        monkeypatch.setitem(CODE_NAME_REGISTRY, "helena", slow_builder)
        set_checkpointer(MagicMock())

        results = await asyncio.gather(
            get_or_compile_graph("helena"),
            get_or_compile_graph("helena"),
            get_or_compile_graph("helena"),
            get_or_compile_graph("helena"),
        )
        assert all(r is sentinel for r in results)
        assert call_count["n"] == 1


class TestWarmUp:
    @pytest.mark.asyncio
    async def test_compiles_subset(self, monkeypatch):
        compiled: list[str] = []

        def tracking_builder(name):
            def _b(_ckpt):
                compiled.append(name)
                return f"graph-{name}"

            return _b

        for code_name in ("helena", "sofia", "giulia", "marco"):
            monkeypatch.setitem(
                CODE_NAME_REGISTRY, code_name, tracking_builder(code_name)
            )

        ck = MagicMock()
        result = await warm_up(["helena", "sofia"], ck)

        assert sorted(result) == ["helena", "sofia"]
        assert sorted(compiled) == ["helena", "sofia"]
        # giulia and marco were not warmed up
        assert "giulia" not in compiled
        assert "marco" not in compiled

    @pytest.mark.asyncio
    async def test_skips_unknown_code_names(self, monkeypatch):
        compiled: list[str] = []

        monkeypatch.setitem(
            CODE_NAME_REGISTRY, "helena",
            lambda _ckpt: compiled.append("helena") or "g",
        )
        ck = MagicMock()
        result = await warm_up(["helena", "ghost"], ck)
        assert result == ["helena"]
        assert compiled == ["helena"]
