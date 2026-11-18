defmodule Dsex.MultiChainComparison do
  @moduledoc """
  Wraps `dspy.MultiChainComparison`: compares `m` parallel reasoning
  attempts and asks the LM to holistically pick the best one. Its `forward/1`
  takes the signature's normal inputs plus a `completions:` list of
  attempt maps (one per attempt, each shaped like the plain signature's
  output fields, plus an optional `rationale`/`reasoning`).
  """

  @doc """
  Options: `:m` (number of attempts to compare, default `3`), plus any
  `dspy.Predict` config (e.g. `temperature:`).

  ## Examples

      iex> program = Dsex.MultiChainComparison.new("question -> answer", m: 3)
      iex> match?(%Dsex.Program{}, program)
      true
  """
  @spec new(term(), keyword()) :: Dsex.Program.t()
  def new(signature, opts \\ []) do
    sig = Dsex.Signature.from(signature)
    {m, opts} = Keyword.pop(opts, :m, 3)
    Dsex.Python.call("dspy.MultiChainComparison", [sig], Keyword.put(opts, :M, m))
  end
end
