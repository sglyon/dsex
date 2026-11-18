defmodule Dsex.Optimizers.BootstrapFewShot do
  @moduledoc """
  Bootstraps demos by running a teacher (the student, if none is given) over
  the trainset and keeping only the examples the metric accepts, falling back
  to labeled demos to fill the rest.

  Wraps `dspy.BootstrapFewShot`. Options: `:metric` (Elixir fn/2 or /3 — the
  third argument is a `trace` during bootstrapping — or any callable; may be
  omitted to accept every bootstrapped example), `:metric_threshold`,
  `:teacher_settings`, `:max_bootstrapped_demos` (default 4),
  `:max_labeled_demos` (default 16), `:max_rounds` (default 1), `:max_errors`.

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `teacher:`
  (defaults to a copy of the student).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> metric = fn example, pred -> example.answer == pred.answer end
      iex> Dsex.Optimizers.BootstrapFewShot.new(metric: metric, max_bootstrapped_demos: 2)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts \\ []) do
    Optimizer.new("dspy.BootstrapFewShot", opts)
  end
end
