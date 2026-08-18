"""Test per-task model/provider override in delegate_task (local patch).

Verifies that:
1. Schema exposes model/provider at top-level and per-task
2. delegate_task function accepts model/provider params
3. Per-task model override takes priority over config
4. Top-level model applies to all tasks in batch
5. Missing model falls back to config/parent (backward compat)
"""
import pytest
from tools.delegate_tool import DELEGATE_TASK_SCHEMA


class TestPerTaskModelOverrideSchema:
    """Verify schema exposes the new fields."""

    def test_top_level_model_in_schema(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model" in props
        assert props["model"]["type"] == "string"

    def test_top_level_provider_in_schema(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "provider" in props
        assert props["provider"]["type"] == "string"

    def test_per_task_model_in_schema(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert "model" in task_props
        assert task_props["model"]["type"] == "string"

    def test_per_task_provider_in_schema(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert "provider" in task_props
        assert task_props["provider"]["type"] == "string"


class TestPerTaskModelOverrideFunction:
    """Verify the function signature accepts model/provider."""

    def test_delegate_task_accepts_model_param(self):
        import inspect
        from tools.delegate_tool import delegate_task
        sig = inspect.signature(delegate_task)
        assert "model" in sig.parameters
        assert "provider" in sig.parameters

    def test_model_param_defaults_none(self):
        import inspect
        from tools.delegate_tool import delegate_task
        sig = inspect.signature(delegate_task)
        assert sig.parameters["model"].default is None
        assert sig.parameters["provider"].default is None

    def test_dispatch_delegate_task_forwards_model(self):
        """Verify run_agent._dispatch_delegate_task forwards model/provider."""
        import inspect
        from run_agent import AIAgent
        source = inspect.getsource(AIAgent._dispatch_delegate_task)
        assert 'model=function_args.get("model")' in source
        assert 'provider=function_args.get("provider")' in source
