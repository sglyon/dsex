defmodule Dsex.ChainOfThought do
  @moduledoc """
  Wraps `dspy.ChainOfThought`: a `Dsex.Predict` that reasons (a `:reasoning`
  output field) before producing the signature's declared outputs.
  """

  @doc """
  Builds a `dspy.ChainOfThought` program for `signature`. `opts` are
  forwarded to the underlying `dspy.Predict` (e.g. `temperature:`); see
  `dspy.ChainOfThought` for `rationale_field:`/`rationale_field_type:`.

  ## Examples

      iex> program = Dsex.ChainOfThought.new("question -> answer")
      iex> match?(%Dsex.Program{}, program)
      true
  """
  @spec new(term(), keyword()) :: Dsex.Program.t()
  def new(signature, opts \\ []) do
    sig = Dsex.Signature.from(signature)
    Dsex.Python.call("dspy.ChainOfThought", [sig], opts)
  end
end
