# Dsex

[DSPy](https://dspy.ai) for Elixir. Dsex runs the real DSPy (3.5.0) in an embedded Python
interpreter via [Pythonx](https://hexdocs.pm/pythonx), so every DSPy module, adapter, and
optimizer works as upstream does. Metrics, tools, and program logic can be plain Elixir functions:
DSPy calls back into the BEAM, including from its worker threads.

## Install

```elixir
def deps do
  [{:dsex, path: "../dsex"}]
end
```

The first Dsex call downloads Python and DSPy with uv (cached afterwards). Configure the
environment if needed:

```elixir
config :dsex, :python,
  python: "==3.13.*",
  dependencies: ["dspy[optuna,litellm]==3.5.0"],
  extra_dependencies: []
```

## Quick tour

```elixir
Dsex.configure(lm: Dsex.LM.new("openai/gpt-4o-mini"))

# Or through a LiteLLM proxy such as the Areté gateway. The `litellm_proxy/` prefix keeps
# native reasoning (ChainOfThought's `reasoning`) intact; `openai/` drops it.
Dsex.configure(
  lm:
    Dsex.LM.new("litellm_proxy/anthropic/claude-haiku-4-5",
      api_base: System.fetch_env!("LLM_GATEWAY_URL"),
      api_key: System.fetch_env!("LLM_GATEWAY_KEY")
    )
)

# Signatures: string form or DSL
defmodule QA do
  use Dsex.Signature
  @moduledoc "Answer questions with short factoid answers."
  input :question, :string
  output :answer, :string, desc: "often between 1 and 5 words"
end

qa = Dsex.ChainOfThought.new(QA)
pred = Dsex.call(qa, question: "What is the capital of France?")
Dsex.Prediction.get(pred, :answer)
#=> "Paris"

# Custom programs: Elixir forward, DSPy predictors underneath
defmodule RAG do
  use Dsex.Module
  predictor :generate, Dsex.ChainOfThought, "context, question -> answer"

  def forward(prog, %{question: q}) do
    Dsex.call(prog.generate, context: MySearch.top_k(q), question: q)
  end
end

# Data, metrics, evaluation
trainset =
  for {q, a} <- pairs, do: Dsex.Example.new(%{question: q, answer: a}, inputs: [:question])

metric = fn example, pred ->
  Dsex.Example.get(example, :answer) == Dsex.Prediction.get(pred, :answer)
end

%{score: score} = Dsex.Evaluate.run(RAG.new(), devset, metric, num_threads: 8)

# Optimizers
optimizer = Dsex.Optimizers.MIPROv2.new(metric: metric, auto: "light")
compiled = Dsex.Optimizer.compile(optimizer, RAG.new(), trainset: trainset)
Dsex.Program.save(compiled, "rag.json")
```

## How it works

See [the design doc](docs/superpowers/specs/2026-11-17-dsex-design.md). In short: Elixir calls
Python through `Pythonx.eval`; Elixir functions handed to Python become `ElixirFn` objects that
message `Dsex.Bridge`, which runs the function in a task (inheriting DSPy's trace/context) and
replies to the waiting Python thread.

## Caveats

- One Python interpreter per node; the GIL serializes Python bytecode, but LM HTTP calls release
  it, so DSPy's thread pools still overlap requests.
- Every callback reply needs a free dirty CPU scheduler. Keep `num_threads` below the dirty CPU
  scheduler count (`+SDcpu`) when callbacks themselves call DSPy.
- Programs holding Elixir functions can be saved as JSON state (`Dsex.Program.save/2`), not
  pickled.
