defmodule Dsex.Optimizers.BootstrapFewShotWithRandomSearch do
  @moduledoc """
  Runs `Dsex.Optimizers.BootstrapFewShot` repeatedly with different demo
  samples/seeds and keeps the candidate program with the best validation
  score.

  Wraps `dspy.BootstrapFewShotWithRandomSearch`. Options: `:metric`
  (required), `:teacher_settings`, `:max_bootstrapped_demos` (default 4),
  `:max_labeled_demos` (default 16), `:max_rounds` (default 1),
  `:num_candidate_programs` (default 16), `:num_threads`, `:stop_at_score`,
  `:metric_threshold`, `:max_errors`.

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `teacher:`,
  `valset:`, `restrict:`, `labeled_sample:` (default `true`).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> metric = fn example, pred -> example.answer == pred.answer end
      iex> Dsex.Optimizers.BootstrapFewShotWithRandomSearch.new(metric: metric, num_candidate_programs: 2)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts) do
    Optimizer.new("dspy.BootstrapFewShotWithRandomSearch", opts)
  end
end
