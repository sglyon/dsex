defmodule Dsex.BestOfN do
  @moduledoc """
  Wraps `dspy.BestOfN`: runs a module up to `N` times at `temperature: 1.0`
  and returns the highest-reward prediction (or the first scoring at or
  above `threshold`).
  """

  @doc """
  Options: `:n` (max attempts), `:reward_fn` (a function receiving the call's
  input map and the resulting `%Dsex.Prediction{}`, returning a numeric
  reward), `:threshold` (an attempt scoring at or above this returns
  immediately), `:fail_count` (allowed failures before raising; defaults to
  `:n`).

  ## Examples

      iex> qa = Dsex.ChainOfThought.new("question -> answer")
      iex> reward = fn _inputs, pred ->
      ...>   if Dsex.Prediction.get(pred, :answer) == "Paris", do: 1.0, else: 0.0
      ...> end
      iex> program = Dsex.BestOfN.new(qa, n: 1, reward_fn: reward, threshold: 1.0)
      iex> match?(%Dsex.Program{}, program)
      true
  """
  @spec new(Dsex.Program.t(), keyword()) :: Dsex.Program.t()
  def new(program, opts) do
    kwargs =
      [
        module: program,
        N: Keyword.fetch!(opts, :n),
        reward_fn: Keyword.fetch!(opts, :reward_fn),
        threshold: Keyword.fetch!(opts, :threshold)
      ]
      |> maybe_put(:fail_count, opts[:fail_count])

    Dsex.Python.call("dspy.BestOfN", [], kwargs)
  end

  defp maybe_put(kwargs, _key, nil), do: kwargs
  defp maybe_put(kwargs, key, value), do: Keyword.put(kwargs, key, value)
end
