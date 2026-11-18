defmodule Dsex.Optimizers.COPRO do
  @moduledoc """
  Coordinate ascent over instructions: repeatedly asks a prompt LM to propose
  new instructions (and output-field prefixes) for each predictor, scores
  every candidate on the trainset, and keeps the best-performing ones.

  Wraps `dspy.COPRO`. Options: `:prompt_model` (defaults to the configured
  LM), `:metric` (required), `:breadth` (new prompts proposed per iteration;
  default 10, must be `> 1`), `:depth` (proposal iterations; default 3),
  `:init_temperature` (default 1.4), `:track_stats` (default `false`).

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `eval_kwargs:`
  (map/keyword forwarded to the internal `dspy.Evaluate`, e.g.
  `num_threads:`, `display_progress:`).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> metric = fn example, pred -> example.answer == pred.answer end
      iex> Dsex.Optimizers.COPRO.new(metric: metric, breadth: 2, depth: 1)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts) do
    Optimizer.new("dspy.COPRO", opts)
  end
end
