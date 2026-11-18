defmodule Dsex.Convert do
  @moduledoc """
  Conversion between Elixir terms and Python objects.

  Elixir → Python goes through `Pythonx.Encoder`: built-in terms map to their
  Python counterparts (atoms other than `nil`/booleans become strings), Dsex
  wrapper structs pass their Python object, and functions become
  `dsex_bridge.ElixirFn` callables (see `Dsex.Bridge`).

  Python → Elixir: the bridge's `export()` marks DSPy objects, which
  `from_py/1` turns into `Dsex.Example`, `Dsex.Prediction`, `Dsex.Program`,
  `Dsex.Signature`, and `Dsex.LM` structs. Other non-primitive objects stay
  `Pythonx.Object`s.
  """

  @wrappers %{
    "example" => Dsex.Example,
    "prediction" => Dsex.Prediction,
    "program" => Dsex.Program,
    "signature" => Dsex.Signature,
    "lm" => Dsex.LM
  }

  @doc "Decodes a `Pythonx.Object` (or an already-decoded term) produced by `export()`."
  @spec from_py(term()) :: term()
  def from_py(%Pythonx.Object{} = object), do: object |> Pythonx.decode() |> walk()
  def from_py(term), do: walk(term)

  defp walk({"__dsex__", "object", %Pythonx.Object{} = object}), do: object

  defp walk({"__dsex__", kind, %Pythonx.Object{} = object}),
    do: struct!(Map.fetch!(@wrappers, kind), py: object)

  defp walk(list) when is_list(list), do: Enum.map(list, &walk/1)
  defp walk(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> walk() |> List.to_tuple()

  defp walk(%Pythonx.Object{} = object), do: object

  defp walk(map) when is_map(map) and not is_struct(map),
    do: Map.new(map, fn {k, v} -> {walk(k), walk(v)} end)

  defp walk(other), do: other

  @doc """
  Encodes `term` as a Python object.

  Unlike `Pythonx.encode!/1`, valid UTF-8 binaries become `str` (DSPy works
  with text); other binaries become `bytes`. Everything else defers to
  `Pythonx.Encoder`.
  """
  @spec to_py(term()) :: Pythonx.Object.t()
  def to_py(term), do: Pythonx.encode!(term, &encode/2)

  @doc false
  def encode(binary, _encoder) when is_binary(binary) do
    if String.valid?(binary),
      do: Pythonx.NIF.unicode_from_string(binary),
      else: Pythonx.NIF.bytes_from_binary(binary)
  end

  def encode(term, encoder), do: Pythonx.Encoder.encode(term, encoder)

  @doc """
  Encodes keyword lists or maps as a Python `dict` with string keys, for use
  as `**kwargs`.
  """
  @spec to_py_kwargs(map() | keyword() | nil) :: Pythonx.Object.t()
  def to_py_kwargs(nil), do: to_py(%{})

  def to_py_kwargs(kwargs) when is_list(kwargs) or is_map(kwargs) do
    kwargs |> Map.new(fn {k, v} -> {to_string(k), v} end) |> to_py()
  end

  @doc "Converts string keys to atoms (used for keyword-style inputs from Python)."
  @spec atomize_keys(map()) :: map()
  def atomize_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {String.to_atom(k), v}
      {k, v} -> {k, v}
    end)
  end
end

defimpl Pythonx.Encoder, for: Function do
  def encode(fun, _encoder), do: Dsex.Bridge.to_python(fun)
end
