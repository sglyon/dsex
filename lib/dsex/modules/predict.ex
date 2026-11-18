defmodule Dsex.Predict do
  @moduledoc """
  Wraps `dspy.Predict`: the base module that maps signature inputs to
  outputs with a single LM call.
  """

  @doc """
  Builds a `dspy.Predict` program for `signature` (any form
  `Dsex.Signature.from/1` accepts: a string, a `use Dsex.Signature` module,
  or a `%Dsex.Signature{}`). `opts` are default LM config forwarded to every
  call (e.g. `temperature:`), overridable per call via DSPy's `config:`
  kwarg.

  ## Examples

      iex> program = Dsex.Predict.new("question -> answer")
      iex> match?(%Dsex.Program{}, program)
      true
  """
  @spec new(term(), keyword()) :: Dsex.Program.t()
  def new(signature, opts \\ []) do
    sig = Dsex.Signature.from(signature)
    Dsex.Python.call("dspy.Predict", [sig], opts)
  end
end
