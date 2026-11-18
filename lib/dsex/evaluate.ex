defmodule Dsex.Evaluate do
  @moduledoc """
  Wraps `dspy.Evaluate`: scores a program over a devset with a metric,
  optionally in parallel.
  """

  alias Dsex.Python

  @typedoc "One devset example scored against the program's prediction for it."
  @type result :: {Dsex.Example.t(), Dsex.Prediction.t(), number()}

  @doc """
  Runs `program` over `devset`, scoring each prediction with `metric`.

  `metric` is an Elixir 2-arity function `fn example, prediction -> ... end`
  (DSPy also calls it with a third `trace` argument during bootstrapping;
  Dsex.Bridge pads/truncates automatically) returning a boolean or number, or
  any Python callable — e.g. one returned by `Dsex.Metrics`.

  `opts` (all optional, forwarded to `dspy.Evaluate`): `:num_threads`,
  `:display_progress` (default `false`), `:display_table`, `:max_errors`,
  `:failure_score`.

  Returns `%{score: float, results: [{example, prediction, score}]}`, where
  `score` is the overall percentage (0–100) and each result tuple's score is
  the raw per-example metric output.

  ## Examples

      iex> Dsex.Evaluate.run(program, devset, fn ex, pred -> ex.answer == pred.answer end)
      %{score: 100.0, results: [{%Dsex.Example{}, %Dsex.Prediction{}, true}]}

      iex> Dsex.Evaluate.run(program, devset, Dsex.Metrics.answer_exact_match())
      %{score: 100.0, results: [{%Dsex.Example{}, %Dsex.Prediction{}, true}]}
  """
  @spec run(Dsex.Program.t(), [Dsex.Example.t()], term(), keyword() | map()) ::
          %{score: float(), results: [result()]}
  def run(program, devset, metric, opts \\ []) when is_list(devset) do
    kwargs = Enum.into(opts, %{devset: devset, metric: metric})
    evaluator = Python.call("dspy.Evaluate", [], kwargs)
    result = Python.call_method(evaluator, "__call__", [program])

    %{
      score: Python.getattr(result, "score"),
      results: Python.getattr(result, "results")
    }
  end
end
