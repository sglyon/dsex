# Executed once by Dsex.Python.init/1 inside a Pythonx.eval call, with
# `bridge_pid` (the Dsex.Bridge process) passed in as a global.
#
# Everything here is defined in that eval's globals on purpose: Pythonx's
# `send_tagged_object` locates its evaluation context by walking the call
# stack for frames whose globals contain `__pythonx_eval_info_bytes__`.
# Functions defined here keep those globals, which lets callbacks fire from
# DSPy worker threads (Evaluate, optimizers) that have no Pythonx frame.
import queue
import signal as _signal
import sys
import types

import pythonx

import dspy
import dspy.utils.parallelizer as _parallelizer
from dspy.dsp.utils.settings import thread_local_overrides

# `dspy.dsp.utils.settings` the attribute is the Settings instance; we need the module.
_settings = sys.modules["dspy.dsp.utils.settings"]

# Pythonx routes stdout/stderr writes and send_tagged_object through
# get_eval_info_bytes(), which finds the active Pythonx.eval by walking the
# stack and raises StopIteration on threads that never entered one (tqdm
# progress bars in GEPA's/Evaluate's worker threads). Fall back to this init
# eval's info: off its scheduler thread Pythonx then sends with a NULL env,
# which is the documented path for non-scheduler threads.
_pythonx_globals = type(sys.stdout).write.__globals__
_pythonx_get_eval_info_bytes = _pythonx_globals["get_eval_info_bytes"]
_fallback_eval_info_bytes = _pythonx_get_eval_info_bytes()


def _get_eval_info_bytes():
    try:
        return _pythonx_get_eval_info_bytes()
    except StopIteration:
        return _fallback_eval_info_bytes


_pythonx_globals["get_eval_info_bytes"] = _get_eval_info_bytes

# DSPy's ParallelExecutor swaps the process SIGINT handler while it runs. Under
# the BEAM, Python never installed a SIGINT handler (getsignal() returns None,
# which cannot be restored) and the BEAM owns SIGINT. Make the swap a no-op.
_parallelizer.signal = types.SimpleNamespace(
    SIGINT=_signal.SIGINT,
    getsignal=lambda _sig: _signal.SIG_DFL,
    signal=lambda _sig, _handler: None,
    default_int_handler=_signal.default_int_handler,
)


class ElixirError(RuntimeError):
    """Raised in Python when an Elixir callback raises or exits."""


class Ref:
    """Opaque holder: Pythonx.decode/1 leaves unknown types as objects, so a Ref
    survives decoding with the wrapped object's identity intact."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj


def export(value):
    """Prepare a value for Elixir: containers and primitives decode natively,
    DSPy objects become ("__dsex__", kind, obj) triples and anything else an
    ("__dsex__", "object", obj) triple. Pythonx.decode/1 leaves those objects
    undecoded, so Elixir receives them by reference. dspy.Reasoning (the type
    of ChainOfThought's `reasoning`) is text to callers, so it exports as str."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dspy.Reasoning):
        return str(value)
    if isinstance(value, dspy.Prediction):
        return ("__dsex__", "prediction", value)
    if isinstance(value, dspy.Example):
        return ("__dsex__", "example", value)
    if isinstance(value, dspy.Module):
        return ("__dsex__", "program", value)
    if isinstance(value, type) and issubclass(value, dspy.Signature):
        return ("__dsex__", "signature", value)
    if isinstance(value, dspy.BaseLM):
        return ("__dsex__", "lm", value)
    if isinstance(value, list):
        return [export(v) for v in value]
    if isinstance(value, tuple):
        return tuple(export(v) for v in value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: export(v) for k, v in value.items()}
    return ("__dsex__", "object", value)


def unref(value):
    return value.obj if isinstance(value, Ref) else value


class ElixirFn:
    """A Python callable backed by an Elixir function registered in Dsex.Bridge."""

    def __init__(self, fid, name=None, doc=None):
        self.fid = fid
        self.__name__ = name or f"elixir_fn_{fid}"
        self.__qualname__ = self.__name__
        self.__doc__ = doc

    def __call__(self, *args, **kwargs):
        reply = queue.Queue(maxsize=1)
        overrides = dict(thread_local_overrides.get())
        pythonx.send_tagged_object(
            bridge_pid,
            "dsex_call",
            (self.fid, export(list(args)), export(kwargs), Ref(overrides), reply),
        )
        status, value = reply.get()
        if status == "ok":
            return value
        raise ElixirError(value)

    # DSPy deep-copies programs and optimizer configs; the Elixir function is
    # immutable, so copies share the registration.
    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def __reduce__(self):
        raise TypeError("ElixirFn objects cannot be pickled")

    def __del__(self):
        try:
            pythonx.send_tagged_object(bridge_pid, "dsex_release", self.fid)
        except Exception:
            pass

    def __repr__(self):
        return f"<ElixirFn {self.__name__}>"


def invoke(base, ctx, target, args, kwargs):
    """Call `target(*args, **kwargs)` and return the result prepared for Elixir.

    `base` is None or a Ref to the dspy.context overrides captured by an ElixirFn
    callback (so nested calls see the caller's trace, LM, ...). `ctx` holds the
    overrides from Dsex.context/2 and wins over `base`. `target` is a dotted
    path string or a callable."""
    overrides = {**(unref(base) or {}), **(ctx or {})}
    if isinstance(target, str):
        target = resolve(target)
    if overrides:
        with dspy.context(**overrides):
            return export(target(*args, **kwargs))
    return export(target(*args, **kwargs))


def resolve(path):
    """Resolve a dotted path such as 'dspy.ChainOfThought'."""
    module_name, _, attr = path.rpartition(".")
    __import__(module_name)
    return getattr(sys.modules[module_name], attr)


def configure(**kwargs):
    """dspy.configure without its owner-thread check: Pythonx runs each eval on
    whichever dirty scheduler thread is free, so no thread can own the config."""
    with _settings.global_lock:
        for key, value in kwargs.items():
            _settings.main_thread_config[key] = value


dsex_bridge = types.ModuleType("dsex_bridge")
for _name in (
    "ElixirError",
    "ElixirFn",
    "Ref",
    "configure",
    "export",
    "invoke",
    "resolve",
    "unref",
):
    setattr(dsex_bridge, _name, globals()[_name])
sys.modules["dsex_bridge"] = dsex_bridge
