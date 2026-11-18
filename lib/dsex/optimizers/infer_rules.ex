defmodule Dsex.Optimizers.InferRules do
  @moduledoc """
  Bootstraps demos like `Dsex.Optimizers.BootstrapFewShot`, then asks the LM
  to induce natural-language rules from the trainset and appends them to each
  optimized predictor's instructions.

  Wraps `dspy.InferRules`, a `dspy.BootstrapFewShot` subclass. Options:
  `:num_candidates` (rule sets to try; default 10), `:num_rules` (per
  candidate; default 10), `:num_threads`, `:teacher_settings`, plus any
  `Dsex.Optimizers.BootstrapFewShot` option (`:metric`,
  `:max_bootstrapped_demos`, `:max_labeled_demos`, `:max_rounds`,
  `:max_errors`).

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `valset:`
  (defaults to half of `trainset`, held out).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> metric = fn example, pred -> example.answer == pred.answer end
      iex> Dsex.Optimizers.InferRules.new(metric: metric, num_candidates: 2, num_rules: 1)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts \\ []) do
    Optimizer.new("dspy.InferRules", opts)
  end
end
