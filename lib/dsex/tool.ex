defmodule Dsex.Tool do
  @moduledoc """
  Wraps an Elixir function as a `dspy.Tool`, for `Dsex.ReAct`'s `tools:`
  option or any other DSPy API that accepts `dspy.Tool` instances.

  `dspy.Tool` normally infers a function's argument schema from its type
  hints (see `dspy/adapters/types/tool.py`'s `_parse_function`). An Elixir
  function arrives in Python as `ElixirFn.__call__(self, *args, **kwargs)`,
  so that inference would produce a bogus `args`/`kwargs` schema; this module
  always passes `args:` explicitly instead of leaving it to be inferred.
  """

  use Dsex.Wrapper

  @typedoc "A bare function, `{name, doc, fun}`, or an already-built `%Dsex.Tool{}`."
  @type spec :: function() | {String.t(), String.t(), function()} | t()

  @doc """
  Wraps `fun` as a `dspy.Tool`. DSPy calls `fun.(**kwargs)`; per
  `Dsex.Bridge`'s arity adaptation, a 1-arity Elixir function receives the
  tool's keyword arguments as a single atom-keyed map.

  Options:

    * `:name` — the tool's name, shown to the LM (defaults to the
      function's auto-generated `ElixirFn` name if omitted).
    * `:desc` — a description of what the tool does.
    * `:args` — a map from argument name to a JSON schema map (e.g.
      `%{city: %{type: "string"}}`) or a bare type atom (e.g. `%{city: :string}`,
      shorthand for `%{type: "string"}`). Defaults to `%{}`, which accepts
      any keyword argument without validation.

  ## Examples

      iex> weather = fn %{city: city} -> "sunny in " <> city end
      iex> tool = Dsex.Tool.new(weather,
      ...>   name: "get_weather",
      ...>   desc: "Look up the current weather for a city.",
      ...>   args: %{city: :string}
      ...> )
      iex> match?(%Dsex.Tool{}, tool)
      true
  """
  @spec new(function(), keyword()) :: t()
  def new(fun, opts \\ []) when is_function(fun) and is_list(opts) do
    name = opts[:name]
    desc = opts[:desc]
    args = opts |> Keyword.get(:args, %{}) |> normalize_args()

    elixir_fn = Dsex.Bridge.to_python(fun, name: name, doc: desc)
    py = Dsex.Python.call("dspy.Tool", [elixir_fn], name: name, desc: desc, args: args)
    %__MODULE__{py: py}
  end

  @doc """
  Normalizes a `Dsex.ReAct` tool `spec/0` into a `%Dsex.Tool{}`. Bare
  functions and `{name, doc, fun}` tuples get an empty `args:` schema (see
  `new/2`); `index` names an otherwise-anonymous bare function
  `"tool_<index + 1>"`.
  """
  @spec coerce(spec(), non_neg_integer()) :: t()
  def coerce(%__MODULE__{} = tool, _index), do: tool

  def coerce({name, doc, fun}, _index) when is_binary(name) and is_function(fun),
    do: new(fun, name: name, desc: doc)

  def coerce(fun, index) when is_function(fun), do: new(fun, name: "tool_#{index + 1}")

  defp normalize_args(args) when is_map(args) do
    Map.new(args, fn {name, spec} -> {to_string(name), normalize_spec(spec)} end)
  end

  defp normalize_spec(type) when is_atom(type) and not is_nil(type),
    do: %{"type" => Atom.to_string(type)}

  defp normalize_spec(spec) when is_map(spec),
    do: Map.new(spec, fn {key, value} -> {to_string(key), value} end)
end
