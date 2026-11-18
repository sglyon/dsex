defmodule Dsex.Prediction do
  @moduledoc """
  Wraps `dspy.Prediction`, the output container returned by DSPy programs
  and predictors. Supports the same `get/2,3` and `to_map/1` as
  `Dsex.Example` (`Prediction` is a subclass of `Example`), plus access to
  the underlying per-candidate `completions` (set when a predictor produced
  more than one candidate, e.g. with `n > 1`).

  ## Examples

      iex> Dsex.Prediction.to_map(Dsex.Prediction.new(answer: "4"))
      %{answer: "4"}
  """

  use Dsex.Wrapper

  @doc """
  Builds a new `%Dsex.Prediction{}` from `fields` (`dspy.Prediction(**fields)`).

  ## Examples

      iex> Dsex.Prediction.new(answer: "4")
  """
  @spec new(map() | keyword()) :: t()
  def new(fields) when is_map(fields) or is_list(fields) do
    Dsex.Python.call("dspy.Prediction", [], fields)
  end

  @doc """
  Returns the value for `key`, or `default` when the field is absent.

  ## Examples

      iex> Dsex.Prediction.get(Dsex.Prediction.new(answer: "4"), :answer)
      "4"
  """
  @spec get(t(), atom() | String.t(), term()) :: term()
  def get(%__MODULE__{} = prediction, key, default \\ nil) do
    Dsex.Python.call_method(prediction, "get", [Kernel.to_string(key), default])
  end

  @doc """
  Converts `prediction` to a map with atom keys (`dspy.Prediction.toDict/0`).

  ## Examples

      iex> Dsex.Prediction.to_map(Dsex.Prediction.new(answer: "4"))
      %{answer: "4"}
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = prediction) do
    prediction |> Dsex.Python.call_method("toDict", []) |> Dsex.Convert.atomize_keys()
  end

  @doc """
  Returns each per-candidate completion as a map with atom keys, in order.

  When `prediction` has no recorded completions (e.g. it was built directly
  with `new/1` rather than returned by a predictor), returns a
  single-element list built from `to_map/1`.

  ## Examples

      iex> Dsex.Prediction.completions(Dsex.Prediction.new(answer: "4"))
      [%{answer: "4"}]
  """
  @spec completions(t()) :: [map()]
  def completions(%__MODULE__{} = prediction) do
    case Dsex.Python.getattr(prediction, "completions") do
      nil ->
        [to_map(prediction)]

      completions ->
        length = Dsex.Python.call("builtins.len", [completions])

        Enum.map(0..(length - 1), fn index ->
          completions
          |> Dsex.Python.call_method("__getitem__", [index])
          |> to_map()
        end)
    end
  end
end
