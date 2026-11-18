defmodule Dsex.Optimizers.KNNFewShot do
  @moduledoc """
  Attaches the `k` trainset examples nearest the current input (by embedding
  similarity) as demos, then bootstraps from them at call time — the demo set
  is chosen per input rather than once at compile time.

  Wraps `dspy.KNNFewShot`. Options: `:k` (required), `:trainset` (required),
  `:vectorizer` (required) — a `dspy.Embedder`, a `%Pythonx.Object{}`
  embedder, or a plain Elixir function `fn texts -> vectors end` mapping a
  list of input strings to a list of float vectors, which is wrapped in
  `dspy.Embedder/1` automatically. Any other option is forwarded to the
  internal `Dsex.Optimizers.BootstrapFewShot` (`:metric`,
  `:max_bootstrapped_demos`, ...).

  `Dsex.Optimizer.compile/3` option: `teacher:`.
  """

  alias Dsex.Optimizer
  alias Dsex.Python

  @doc """
  ## Examples

      iex> vectorizer = fn texts -> Enum.map(texts, fn t -> [:erlang.byte_size(t) * 1.0] end) end
      iex> Dsex.Optimizers.KNNFewShot.new(k: 2, trainset: trainset, vectorizer: vectorizer)
      %Dsex.Optimizer{}
  """
  @spec new(keyword() | map()) :: Dsex.Optimizer.t()
  def new(opts) do
    opts
    |> Enum.into(%{})
    |> Map.update!(:vectorizer, &embedder/1)
    |> then(&Optimizer.new("dspy.KNNFewShot", &1))
  end

  defp embedder(fun) when is_function(fun, 1), do: Python.call("dspy.Embedder", [fun])
  defp embedder(other), do: other
end
