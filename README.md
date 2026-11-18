# Dsex

[DSPy](https://dspy.ai) for Elixir.

Dsex runs the real DSPy (3.5.0) inside the BEAM through [Pythonx](https://hexdocs.pm/pythonx),
which embeds CPython in the Erlang VM. You write signatures, programs, metrics, and tools in
Elixir; DSPy's modules, adapters, and optimizers do the rest, unchanged.

## Why

DSPy's value is its optimizers (MIPROv2, GEPA, SIMBA, COPRO, BootstrapFewShot, ...), and they
are the hardest part to port: MIPROv2 depends on optuna's Bayesian search, and GEPA lives in a
separate ~24k-line package. A native port would lag upstream forever.

Wrapping DSPy in-process avoids that, provided Python can call back into Elixir. It can: Elixir
functions passed to DSPy become Python callables that message the BEAM, run in an Elixir task,
and hand the result back, even from DSPy's worker threads. The callback inherits DSPy's trace
context, so optimizers can bootstrap demos through programs whose control flow is Elixir.

## Usage

```elixir
def deps do
  [{:dsex, github: "sglyon/dsex"}]
end
```

The first call downloads Python and DSPy with [uv](https://docs.astral.sh/uv) into a cache;
later runs reuse it.

```elixir
Dsex.configure(lm: Dsex.LM.new("openai/gpt-4o-mini", api_key: System.fetch_env!("OPENAI_API_KEY")))

# Signatures: a string, or a module
defmodule QA do
  use Dsex.Signature
  @moduledoc "Answer questions with short factoid answers."
  input :question, :string
  output :answer, :string, desc: "often between 1 and 5 words"
end

pred = Dsex.call(Dsex.ChainOfThought.new(QA), question: "What is the capital of France?")
Dsex.Prediction.get(pred, :answer)      #=> "Paris"
Dsex.Prediction.get(pred, :reasoning)   #=> "The capital of France is Paris..."

# Programs: Elixir control flow over DSPy predictors
defmodule RAG do
  use Dsex.Module
  predictor :generate, Dsex.ChainOfThought, "context, question -> answer"

  def forward(prog, %{question: q}) do
    Dsex.call(prog.generate, context: MySearch.top_k(q), question: q)
  end
end

# Data and metrics
trainset =
  for {q, a} <- pairs, do: Dsex.Example.new(%{question: q, answer: a}, inputs: [:question])

metric = fn example, pred ->
  Dsex.Example.get(example, :answer) == Dsex.Prediction.get(pred, :answer)
end

%{score: score} = Dsex.Evaluate.run(RAG.new(), trainset, metric, num_threads: 4)

# Optimize, then save
optimizer = Dsex.Optimizers.MIPROv2.new(metric: metric, auto: "light")
compiled = Dsex.Optimizer.compile(optimizer, RAG.new(), trainset: trainset)
Dsex.Program.save(compiled, "rag.json")
```

Any model [LiteLLM](https://docs.litellm.ai) supports works. Through a LiteLLM proxy, use the
`litellm_proxy/` prefix (it keeps native reasoning, which the `openai/` route drops):

```elixir
Dsex.LM.new("litellm_proxy/anthropic/claude-haiku-4-5",
  api_base: System.fetch_env!("LITELLM_PROXY_URL"),
  api_key: System.fetch_env!("LITELLM_PROXY_KEY")
)
```

Also available: `Dsex.Predict`, `ReAct` (Elixir functions as tools), `BestOfN`, `Refine`,
`MultiChainComparison`, `Parallel`; `Dsex.Metrics` (exact match, F1, SemanticF1, ...); and
optimizers `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`,
`KNNFewShot`, `COPRO`, `MIPROv2`, `SIMBA`, `GEPA`, `InferRules`, `Ensemble`. Anything else in
DSPy is reachable through `Dsex.Python.call/3`.

### Configuration

```elixir
config :dsex, :python,
  python: "==3.13.*",
  dependencies: ["dspy[optuna,litellm]==3.5.0"],
  extra_dependencies: []
```

### Caveats

- One Python interpreter per node. The GIL serializes Python bytecode, but LM HTTP calls release
  it, so DSPy's thread pools still overlap requests.
- Each Python call and each callback reply holds a dirty CPU scheduler. If every one is held by
  a call waiting on an Elixir callback, the VM stalls. Keep `num_threads` below the dirty CPU
  scheduler count, or raise it with `+SDcpu`.
- Programs holding Elixir functions save as JSON state (`Dsex.Program.save/2`), not pickles.

## Development

Requires Erlang/OTP 28 and Elixir 1.19 (`mise install` reads `mise.toml`). uv is fetched by
Pythonx; nothing else to install.

```sh
mix deps.get
mix test                              # offline: tests use DSPy's DummyLM
mix format --check-formatted
mix compile --warnings-as-errors
```

Layout:

| Path | What |
|---|---|
| `lib/dsex/python.ex`, `convert.ex` | Interpreter setup, calls, Elixir ⇄ Python conversion |
| `lib/dsex/bridge.ex`, `priv/python/dsex_bridge.py` | Python → Elixir callbacks |
| `lib/dsex/module.ex`, `priv/python/dsex_module.py` | `use Dsex.Module` |
| `lib/dsex/{signature,example,prediction,lm,program}.ex` | Core wrappers |
| `lib/dsex/modules/`, `lib/dsex/optimizers/` | DSPy modules and optimizers |

Design notes: [`docs/superpowers/specs/2026-11-17-dsex-design.md`](docs/superpowers/specs/2026-11-17-dsex-design.md).

To run against a real model, export your provider key and use `mix run` with a script like the
usage example above. Never commit keys; `.env*` files are gitignored.
