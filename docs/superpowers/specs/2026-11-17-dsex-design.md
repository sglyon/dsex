# Dsex: DSPy for Elixir — design

## Intent

Use DSPy's programming model (signatures, modules, metrics, evaluation, optimizers) from Elixir,
with metrics, tools, and custom program logic written in Elixir.

## Approaches considered

1. **Wrap the real DSPy via Pythonx (chosen).** CPython embedded in the BEAM; DSPy
   3.5.0 installed by uv on first use. Every optimizer (MIPROv2, GEPA, SIMBA, COPRO, ...) works
   unchanged and tracks upstream. Cost: the GIL serializes Python bytecode, though DSPy's LM
   calls release it during HTTP I/O, so DSPy's own thread pools still overlap requests.
   `Pythonx.remote_eval/4` + FLAME is the scale-out path if one interpreter saturates.
2. Native Elixir reimplementation. Idiomatic and GIL-free, but MIPROv2 needs a TPE port and GEPA
   needs the separate `gepa` package (~24k LOC) ported; permanent drift from upstream.
3. Rust (DSRs) + Rustler. Pre-1.0 port, and callbacks into Elixir across a NIF boundary are
   harder than with Pythonx.

## Architecture

```
Elixir caller ──Pythonx.eval──▶ dsex_bridge.invoke(ctx, target, args, kwargs) ──▶ DSPy
      ▲                                                                           │
      │  {:dsex_call, (fid, args, kwargs, overrides, reply_queue)}                │
Dsex.Bridge ◀──pythonx.send_tagged_object── ElixirFn.__call__ (any Python thread) ◀┘
      │ runs fun in Task with caller's dspy.context overrides
      └──Pythonx.eval("reply.put(result)")──▶ unblocks waiting Python thread
```

- `Dsex.Python` — init (uv env, bridge load), `call/3`, `call_method/4`, `getattr/2`, `eval/2`.
- `Dsex.Convert` — UTF-8 binaries → `str`; wrapper structs → their object; functions →
  `ElixirFn`. Results: DSPy objects → `Dsex.Example/Prediction/Program/Signature/LM` structs.
- `Dsex.Bridge` — callback registry (ETS), runs callbacks with arity adaptation, propagates
  `dspy.context` overrides (trace!) so DSPy calls made from Elixir callbacks are traced.
- `dsex_bridge.configure` bypasses `dspy.configure`'s owner-thread check (evals run on
  arbitrary dirty-scheduler threads).
- DSPy's SIGINT handler swap in `ParallelExecutor` is disabled: the BEAM owns SIGINT.

## Constraints and risks

- Dirty CPU schedulers bound concurrent evals. A callback reply needs a free dirty scheduler; if
  all are held by evals blocked on Elixir callbacks, progress stalls. Keep `num_threads` below
  the dirty CPU scheduler count (`+SDcpu`) when callbacks themselves call DSPy.
- One interpreter per node; `dspy.configure` is node-global.
- Elixir functions passed to Python cannot be pickled (`save_program=True` unsupported for
  programs holding them); JSON `save/load` of state works.

## Scope

Signatures (string + DSL), Example/Prediction, LM, Predict/ChainOfThought/ReAct/BestOfN/Refine/
MultiChainComparison/Parallel, custom Elixir modules (`use Dsex.Module`), Evaluate, metrics,
all prompt optimizers (LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch,
KNNFewShot, COPRO, MIPROv2, SIMBA, GEPA, Ensemble, InferRules), save/load. Weight
fine-tuning optimizers are reachable through the generic `Dsex.Optimizer.new/2` only.
