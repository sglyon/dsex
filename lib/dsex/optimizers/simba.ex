defmodule Dsex.Optimizers.SIMBA do
  @moduledoc """
  Stochastic Introspective Mini-Batch Ascent: samples mini-batches, finds
  examples where the student's outputs vary a lot across tries, and either
  appends a successful one as a demo or asks the LM to write a
  self-reflective rule.

  Wraps `dspy.SIMBA`. Options: `:metric` (required; Elixir fn/2 returning a
  score, or any callable), `:bsize` (mini-batch size; default 32),
  `:num_candidates` (default 6), `:max_steps` (default 8), `:max_demos`
  (default 4), `:prompt_model`, `:teacher_settings`,
  `:demo_input_field_maxlen` (default 100_000), `:num_threads`,
  `:temperature_for_sampling` (default 0.2), `:temperature_for_candidates`
  (default 0.2).

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `seed:`
  (default 0).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> metric = fn example, pred -> if example.answer == pred.answer, do: 1.0, else: 0.0 end
      iex> Dsex.Optimizers.SIMBA.new(metric: metric, bsize: 2, num_candidates: 2, max_steps: 2)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts) do
    Optimizer.new("dspy.SIMBA", opts)
  end
end
