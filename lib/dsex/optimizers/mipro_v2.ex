defmodule Dsex.Optimizers.MIPROv2 do
  @moduledoc """
  Jointly searches over instructions and few-shot demos with Bayesian
  optimization (via the optional `optuna` package).

  Wraps `dspy.MIPROv2`. Options: `:metric` (required), `:prompt_model`,
  `:task_model` (both default to `dspy.settings.lm` — configure one via
  `Dsex.configure/1` or `Dsex.context/2`, or pass both explicitly),
  `:teacher_settings`, `:max_bootstrapped_demos` (default 4),
  `:max_labeled_demos` (default 4), `:auto` (`"light"`, `"medium"`,
  `"heavy"`, or `nil`; default `"light"`), `:num_candidates` (required when
  `auto: nil`), `:num_threads`, `:seed` (default 9), `:init_temperature`
  (default 1.0), `:track_stats` (default `true`), `:log_dir`,
  `:metric_threshold`, `:max_errors`.

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `valset:`,
  `teacher:`, `num_trials:` (required when `auto: nil`), `minibatch:`
  (default `true`), `minibatch_size:` (default 35),
  `program_aware_proposer:`, `data_aware_proposer:`, `tip_aware_proposer:`,
  `fewshot_aware_proposer:`.

  `compile/3` uses the optional `optuna` Python package for Bayesian search
  (part of Dsex's default `dspy[optuna]` dependency).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> metric = fn example, pred -> example.answer == pred.answer end
      iex> Dsex.Optimizers.MIPROv2.new(metric: metric, auto: nil, num_candidates: 2)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts) do
    Optimizer.new("dspy.MIPROv2", opts)
  end
end
