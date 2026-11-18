# Executed by Dsex.Python after dsex_bridge.py. Defines `ElixirModule`, a
# `dspy.Module` whose sub-programs are real attributes (so `named_predictors`,
# `deepcopy`, `reset_copy`, `save`, and optimizers all see and mutate them
# normally) and whose `forward` delegates to an Elixir callback.
import inspect
import sys
import types

import dspy

# `dspy.Refine` calls `inspect.getsource(reward_fn)` (falling back to
# `inspect.getsource(reward_fn.__class__)`) purely to record a human-readable
# `reward_fn_code` on itself; Elixir reward functions arrive as `ElixirFn`
# instances defined inside a Pythonx eval with no real source file, so both
# attempts raise. Narrowly patch `inspect.getsource` to fall back to a
# placeholder only for dsex's own unintrospectable callables (`ElixirFn` and
# `ElixirModule`, or instances thereof); every other caller still sees the
# original stdlib behavior, including its exceptions.
_original_getsource = inspect.getsource


def _dsex_classes():
    classes = [ElixirModule]
    elixir_fn = getattr(sys.modules.get("dsex_bridge"), "ElixirFn", None)
    if elixir_fn is not None:
        classes.append(elixir_fn)
    return tuple(classes)


def _is_dsex_object(object):
    classes = _dsex_classes()
    if inspect.isclass(object):
        return issubclass(object, classes)
    return isinstance(object, classes)


def _getsource_with_elixir_fallback(object):
    try:
        return _original_getsource(object)
    except (OSError, TypeError):
        if _is_dsex_object(object):
            return f"<source unavailable: {object!r} is defined via dsex (Elixir callback)>"
        raise


inspect.getsource = _getsource_with_elixir_fallback


class ElixirModule(dspy.Module):
    """A `dspy.Module` whose forward logic and named sub-programs come from
    Elixir (see `Dsex.Module`).

    `predictors` is a `dict[str, dspy.Module]`: each entry becomes a real
    instance attribute, exactly as a hand-written `dspy.Module.__init__`
    would assign `self.<name> = <predictor>`. `forward_fn` is called as
    `forward_fn(self, kwargs)` — `self` so the Elixir side can read this
    specific instance's (possibly deep-copied) sub-programs."""

    def __init__(self, predictors, forward_fn):
        super().__init__()
        self._forward_fn = forward_fn
        for name, predictor in predictors.items():
            setattr(self, name, predictor)

    def forward(self, **kwargs):
        return self._forward_fn(self, kwargs)


def new_module(predictors, forward_fn):
    """Builds an `ElixirModule` with `predictors` as its named sub-programs."""
    return ElixirModule(predictors, forward_fn)


dsex_module = types.ModuleType("dsex_module")
for _name in ("ElixirModule", "new_module"):
    setattr(dsex_module, _name, globals()[_name])
sys.modules["dsex_module"] = dsex_module
