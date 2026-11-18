defmodule Dsex.Example do
  @moduledoc """
  Wraps `dspy.Example`, DSPy's dict-like record type used for training/dev
  data and for anything that distinguishes "input" fields from "label"
  fields (metrics, `Evaluate`, bootstrapped demos).

  ## Examples

      iex> example =
      ...>   Dsex.Example.new(%{question: "2+2?", answer: "4"}, inputs: [:question])
      iex> Dsex.Example.to_map(Dsex.Example.inputs(example))
      %{question: "2+2?"}
      iex> Dsex.Example.to_map(Dsex.Example.labels(example))
      %{answer: "4"}
  """

  use Dsex.Wrapper

  @doc """
  Builds a new `%Dsex.Example{}` from `fields` (`dspy.Example(**fields)`).

  `opts[:inputs]`, when given, marks those field names as inputs (as if
  `with_inputs/2` were called on the result).

  ## Examples

      iex> Dsex.Example.new(question: "2+2?", answer: "4")
  """
  @spec new(map() | keyword(), keyword()) :: t()
  def new(fields, opts \\ []) when is_map(fields) or is_list(fields) do
    example = Dsex.Python.call("dspy.Example", [], fields)

    case Keyword.get(opts, :inputs) do
      nil -> example
      keys -> with_inputs(example, keys)
    end
  end

  @doc """
  Returns a new example with `keys` marked as the input fields.

  ## Examples

      iex> Dsex.Example.new(%{question: "2+2?"})
      ...> |> Dsex.Example.with_inputs([:question])
  """
  @spec with_inputs(t(), [atom() | String.t()]) :: t()
  def with_inputs(%__MODULE__{} = example, keys) when is_list(keys) do
    Dsex.Python.call_method(example, "with_inputs", Enum.map(keys, &Kernel.to_string/1))
  end

  @doc """
  Returns the value for `key`, or `default` when the field is absent.

  ## Examples

      iex> Dsex.Example.get(Dsex.Example.new(answer: "4"), :answer)
      "4"
      iex> Dsex.Example.get(Dsex.Example.new(answer: "4"), :missing, "n/a")
      "n/a"
  """
  @spec get(t(), atom() | String.t(), term()) :: term()
  def get(%__MODULE__{} = example, key, default \\ nil) do
    Dsex.Python.call_method(example, "get", [Kernel.to_string(key), default])
  end

  @doc """
  Converts `example` to a map with atom keys (`dspy.Example.toDict/0`).

  ## Examples

      iex> Dsex.Example.to_map(Dsex.Example.new(answer: "4"))
      %{answer: "4"}
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = example) do
    example |> Dsex.Python.call_method("toDict", []) |> Dsex.Convert.atomize_keys()
  end

  @doc "Returns a new example containing only the fields marked as inputs."
  @spec inputs(t()) :: t()
  def inputs(%__MODULE__{} = example), do: Dsex.Python.call_method(example, "inputs", [])

  @doc "Returns a new example containing only the fields *not* marked as inputs."
  @spec labels(t()) :: t()
  def labels(%__MODULE__{} = example), do: Dsex.Python.call_method(example, "labels", [])

  @doc """
  Returns a new example with `key` set to `value`; `example` itself is left
  unchanged (`dspy.Example.copy(**{key: value})`).

  ## Examples

      iex> Dsex.Example.new(answer: "4")
      ...> |> Dsex.Example.put(:answer, "5")
      ...> |> Dsex.Example.get(:answer)
      "5"
  """
  @spec put(t(), atom() | String.t(), term()) :: t()
  def put(%__MODULE__{} = example, key, value) do
    Dsex.Python.call_method(example, "copy", [], %{Kernel.to_string(key) => value})
  end
end
