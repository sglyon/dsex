defmodule Dsex.Optimizers.LabeledFewShot do
  @moduledoc """
  Samples labeled demos straight from the trainset onto a student's
  predictors — the simplest optimizer: no bootstrapping, no extra LM calls.

  Wraps `dspy.LabeledFewShot`. Option: `:k`, demos per predictor (default 16).

  `Dsex.Optimizer.compile/3` options: `trainset:` (required), `sample:`
  (random sample vs. the first `k`; default `true`).
  """

  alias Dsex.Optimizer

  @doc """
  ## Examples

      iex> Dsex.Optimizers.LabeledFewShot.new(k: 4)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts \\ []) do
    Optimizer.new("dspy.LabeledFewShot", opts)
  end
end
