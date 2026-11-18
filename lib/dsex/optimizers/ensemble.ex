defmodule Dsex.Optimizers.Ensemble do
  @moduledoc """
  Wraps several programs so calling the result runs every member and reduces
  their outputs — no training signal needed, so there is nothing to bootstrap
  or score.

  Wraps `dspy.Ensemble`. Options: `:size` (sample this many members per call
  instead of all of them), `:reduce_fn` (an Elixir function or
  `%Pythonx.Object{}` callable taking the list of member outputs; defaults to
  `dspy.majority`, DSPy's vote-by-most-common-completion reducer — pass
  `reduce_fn: nil` to get the raw list of outputs instead).

  `Dsex.Optimizer.compile/3`'s `student` argument here is the *list of
  programs* to ensemble, not a single student.
  """

  alias Dsex.Optimizer
  alias Dsex.Python

  @doc """
  ## Examples

      iex> Dsex.Optimizers.Ensemble.new() |> Dsex.Optimizer.compile(programs)
      %Dsex.Program{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts \\ []) do
    opts
    |> Enum.into(%{})
    |> Map.put_new_lazy(:reduce_fn, &majority/0)
    |> then(&Optimizer.new("dspy.Ensemble", &1))
  end

  defp majority, do: Python.call("dsex_bridge.resolve", ["dspy.majority"])
end
