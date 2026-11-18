defmodule Dsex.Optimizers.GEPA do
  @moduledoc """
  Evolutionary prompt optimizer: reflects on execution traces with a
  `reflection_lm` to propose new predictor instructions, guided by the
  metric's textual feedback.

  Wraps `dspy.GEPA`. Options: `:metric` (required; 5-arity Elixir function
  `fn gold, pred, trace, pred_name, pred_trace -> ... end`, or any callable
  accepting those five positional arguments), `:reflection_lm` (required
  unless `:instruction_proposer` is given), and exactly one budget option —
  `:auto` (`"light"`, `"medium"`, `"heavy"`), `:max_full_evals`, or
  `:max_metric_calls`. Other options: `:reflection_minibatch_size` (default
  3), `:candidate_selection_strategy` (`"pareto"` default, or
  `"current_best"`), `:skip_perfect_score` (default `true`),
  `:instruction_proposer`, `:use_merge` (default `true`),
  `:max_merge_invocations` (default 5), `:num_threads`, `:failure_score`
  (default 0.0), `:perfect_score` (default 1.0), `:track_stats` (default
  `false`), `:track_best_outputs`, `:seed` (default 0).

  An Elixir `:metric` may return a plain number/boolean *or* a map
  `%{score: float, feedback: string}` for per-predictor feedback; the latter
  is converted to `dspy.Prediction(score:, feedback:)`, the shape GEPA expects
  when a metric wants to say more than a bare score. A `%Pythonx.Object{}`
  metric is passed through unchanged.

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `valset:`,
  `teacher:`.
  """

  alias Dsex.Optimizer
  alias Dsex.Prediction

  @doc """
  ## Examples

      iex> metric = fn gold, pred, _trace, _pred_name, _pred_trace ->
      ...>   score = if gold.answer == pred.answer, do: 1.0, else: 0.0
      ...>   %{score: score, feedback: "score was \#{score}"}
      ...> end
      iex> Dsex.Optimizers.GEPA.new(metric: metric, reflection_lm: reflection_lm, max_metric_calls: 6)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts) do
    opts
    |> Enum.into(%{})
    |> Map.update!(:metric, &wrap_metric/1)
    |> then(&Optimizer.new("dspy.GEPA", &1))
  end

  defp wrap_metric(fun) when is_function(fun, 5) do
    fn gold, pred, trace, pred_name, pred_trace ->
      case fun.(gold, pred, trace, pred_name, pred_trace) do
        %{score: score, feedback: feedback} -> Prediction.new(%{score: score, feedback: feedback})
        other -> other
      end
    end
  end

  defp wrap_metric(other), do: other
end
