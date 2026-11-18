defmodule Dsex.Signature do
  @moduledoc """
  Wraps `dspy.Signature`: the typed input/output contract a `Dsex.Program`
  (`Predict`, `ChainOfThought`, ...) reads and writes.

  Build one from a compact string (`"question, context: list[str] -> answer: int"`),
  or declare one as a module:

      defmodule QA do
        use Dsex.Signature
        @moduledoc "Answer the question using the context."

        input :question, :string, desc: "the question to answer"
        input :context, {:list, :string}, desc: "passages to use"
        output :answer, :integer, desc: "the final answer"
      end

      QA.signature()

  `@moduledoc` becomes the signature's `instructions`; omit it to let DSPy
  generate the default "Given the fields ..., produce the fields ..." text.
  `signature/0` is built once per module and cached.

  Every function here that accepts "a signature" takes a string, a module
  using `Dsex.Signature`, or an already-built `%Dsex.Signature{}` -- see
  `from/1`.

  Supported field types: `:string`, `:integer`, `:float`, `:boolean`,
  `:any`, `{:list, type}`, `{:dict, key_type, value_type}`,
  `{:literal, values}`, `{:optional, type}`, or a raw Python type
  expression string (e.g. `"dict[str, list[int]]"`).
  """

  use Dsex.Wrapper
  import Kernel, except: [to_string: 1]

  @doc false
  defmacro __using__(_opts) do
    quote do
      import Dsex.Signature, only: [input: 2, input: 3, output: 2, output: 3]

      Module.register_attribute(__MODULE__, :dsex_fields, accumulate: true)
      @before_compile Dsex.Signature
    end
  end

  @doc """
  Declares an input field named `name` with type `type` inside a module
  using `Dsex.Signature`. `opts` supports `:desc` and `:prefix`.
  """
  defmacro input(name, type, opts \\ []) do
    quote do
      @dsex_fields {:input, unquote(name), unquote(type), unquote(opts)}
    end
  end

  @doc """
  Declares an output field named `name` with type `type` inside a module
  using `Dsex.Signature`. `opts` supports `:desc` and `:prefix`.
  """
  defmacro output(name, type, opts \\ []) do
    quote do
      @dsex_fields {:output, unquote(name), unquote(type), unquote(opts)}
    end
  end

  @doc false
  defmacro __before_compile__(env) do
    fields = env.module |> Module.get_attribute(:dsex_fields) |> Enum.reverse()

    instructions =
      case Module.get_attribute(env.module, :moduledoc) do
        {_line, text} when is_binary(text) -> text
        _other -> nil
      end

    quote do
      @doc "Returns this module's `%Dsex.Signature{}`, built once and cached."
      @spec signature() :: Dsex.Signature.t()
      def signature do
        Dsex.Signature.__cached__(__MODULE__, fn ->
          Dsex.Signature.__build__(
            unquote(Macro.escape(fields)),
            unquote(instructions),
            unquote(Kernel.to_string(env.module))
          )
        end)
      end
    end
  end

  @doc """
  Builds a `%Dsex.Signature{}`.

  `spec` is either a compact string (`"question -> answer"`) wrapping
  `dspy.Signature(spec, instructions)`, or a list of `{:input | :output,
  name, type, opts}` field tuples (the same shape `input/3` and `output/3`
  accumulate), in which case `opts[:instructions]` sets the instructions
  (omit it to use DSPy's default).

  ## Examples

      iex> Dsex.Signature.new("question -> answer")

      iex> Dsex.Signature.new([
      ...>   {:input, :question, :string, []},
      ...>   {:output, :answer, :integer, []}
      ...> ])
  """
  @spec new(String.t() | [tuple()], keyword()) :: t()
  def new(spec, opts \\ [])

  def new(spec, opts) when is_binary(spec) do
    case Keyword.get(opts, :instructions) do
      nil -> Dsex.Python.call("dspy.Signature", [spec])
      instructions -> Dsex.Python.call("dspy.Signature", [spec, instructions])
    end
  end

  def new(fields, opts) when is_list(fields) do
    __build__(fields, Keyword.get(opts, :instructions), "Signature")
  end

  @doc """
  Normalizes `signature` (a string, a module using `Dsex.Signature`, or an
  already-built `%Dsex.Signature{}`) into a `%Dsex.Signature{}`. Every other
  function in this module accepts all three forms directly, via `from/1`.
  """
  @spec from(t() | module() | String.t()) :: t()
  def from(%__MODULE__{} = signature), do: signature
  def from(spec) when is_binary(spec), do: new(spec)

  def from(module) when is_atom(module) do
    Code.ensure_loaded(module)

    if function_exported?(module, :signature, 0) do
      module.signature()
    else
      raise ArgumentError,
            "#{inspect(module)} is not a %Dsex.Signature{} and does not define signature/0"
    end
  end

  @doc "Returns `signature`'s instructions (its DSPy docstring)."
  @spec instructions(t() | module() | String.t()) :: String.t()
  def instructions(signature), do: signature |> from() |> Dsex.Python.getattr("instructions")

  @doc """
  Returns a new signature with the same fields as `signature` and
  instructions replaced by `instructions`.
  """
  @spec with_instructions(t() | module() | String.t(), String.t()) :: t()
  def with_instructions(signature, instructions) do
    signature |> from() |> Dsex.Python.call_method("with_instructions", [instructions])
  end

  @doc """
  Returns `signature`'s input fields, in declaration order, as
  `%{name: atom, type: String.t(), desc: String.t(), prefix: String.t()}`.
  """
  @spec input_fields(t() | module() | String.t()) :: [map()]
  def input_fields(signature), do: fields_of(signature, :input)

  @doc """
  Returns `signature`'s output fields, in declaration order, as
  `%{name: atom, type: String.t(), desc: String.t(), prefix: String.t()}`.
  """
  @spec output_fields(t() | module() | String.t()) :: [map()]
  def output_fields(signature), do: fields_of(signature, :output)

  @doc """
  Returns a new signature with an `:input` or `:output` field named `name`
  appended to the end of its section. `opts` supports `:type` (defaults to
  `:string`), `:desc` and `:prefix`.
  """
  @spec append(t() | module() | String.t(), atom() | String.t(), :input | :output, keyword()) ::
          t()
  def append(signature, name, kind, opts \\ []) when kind in [:input, :output] do
    insert_field(signature, name, kind, opts, "append")
  end

  @doc """
  Returns a new signature with an `:input` or `:output` field named `name`
  inserted at the start of its section. `opts` supports `:type` (defaults
  to `:string`), `:desc` and `:prefix`.
  """
  @spec prepend(t() | module() | String.t(), atom() | String.t(), :input | :output, keyword()) ::
          t()
  def prepend(signature, name, kind, opts \\ []) when kind in [:input, :output] do
    insert_field(signature, name, kind, opts, "prepend")
  end

  @doc """
  Returns `signature`'s compact string form (`"input1, input2 -> output1"`),
  i.e. `dspy.Signature.signature`. This does not include instructions or
  field descriptions; see `instructions/1` and `input_fields/1`.
  """
  @spec to_string(t() | module() | String.t()) :: String.t()
  def to_string(signature), do: signature |> from() |> Dsex.Python.getattr("signature")

  @doc false
  @spec __cached__(module(), (-> t())) :: t()
  def __cached__(module, builder) do
    case :persistent_term.get({__MODULE__, module}, nil) do
      nil ->
        built = builder.()
        :persistent_term.put({__MODULE__, module}, built)
        built

      cached ->
        cached
    end
  end

  @doc false
  @spec __build__([tuple()], String.t() | nil, String.t()) :: t()
  def __build__(fields, instructions, signature_name) do
    specs = Enum.map(fields, &normalize_field/1)

    code = """
    import dspy
    import typing

    fields = {}
    for spec in specs:
        type_ = eval(spec["type"], {"typing": typing})
        field_cls = dspy.InputField if spec["kind"] == "input" else dspy.OutputField
        kwargs = {}
        if spec.get("desc") is not None:
            kwargs["desc"] = spec["desc"]
        if spec.get("prefix") is not None:
            kwargs["prefix"] = spec["prefix"]
        fields[spec["name"]] = (type_, field_cls(**kwargs))

    sig = dspy.make_signature(fields, instructions, signature_name)
    dsex_bridge.export(sig)
    """

    Dsex.Python.eval(code, %{
      "specs" => specs,
      "instructions" => instructions,
      "signature_name" => signature_name
    })
  end

  defp normalize_field({kind, name, type}), do: normalize_field({kind, name, type, []})

  defp normalize_field({kind, name, type, opts}) when kind in [:input, :output] do
    %{
      "name" => Kernel.to_string(name),
      "kind" => Atom.to_string(kind),
      "type" => type_expr(type),
      "desc" => Keyword.get(opts, :desc),
      "prefix" => Keyword.get(opts, :prefix)
    }
  end

  defp insert_field(signature, name, kind, opts, mode) do
    py_signature = from(signature)
    type_spec = Keyword.get(opts, :type, :string)

    code = """
    import dspy
    import typing

    field_cls = dspy.InputField if kind == "input" else dspy.OutputField
    kwargs = {}
    if desc is not None:
        kwargs["desc"] = desc
    if prefix is not None:
        kwargs["prefix"] = prefix

    field = field_cls(**kwargs)
    type_ = eval(type_expr, {"typing": typing})
    new_sig = (
        sig.append(name, field, type_)
        if mode == "append"
        else sig.prepend(name, field, type_)
    )
    dsex_bridge.export(new_sig)
    """

    Dsex.Python.eval(code, %{
      "sig" => py_signature,
      "name" => Kernel.to_string(name),
      "kind" => Atom.to_string(kind),
      "desc" => Keyword.get(opts, :desc),
      "prefix" => Keyword.get(opts, :prefix),
      "type_expr" => type_expr(type_spec),
      "mode" => mode
    })
  end

  defp fields_of(signature, kind) do
    py_signature = from(signature)

    code = """
    import dspy
    from dspy.adapters.utils import get_annotation_name

    attr = "input_fields" if kind == "input" else "output_fields"
    result = []
    for name, field in getattr(sig, attr).items():
        extra = field.json_schema_extra or {}
        result.append({
            "name": name,
            "type": get_annotation_name(field.annotation),
            "desc": extra.get("desc") or "",
            "prefix": extra.get("prefix") or "",
        })
    result
    """

    code
    |> Dsex.Python.eval(%{"sig" => py_signature, "kind" => Atom.to_string(kind)})
    |> Enum.map(fn field ->
      field |> Dsex.Convert.atomize_keys() |> Map.update!(:name, &String.to_atom/1)
    end)
  end

  defp type_expr(spec) when is_binary(spec), do: spec
  defp type_expr(:string), do: "str"
  defp type_expr(:integer), do: "int"
  defp type_expr(:float), do: "float"
  defp type_expr(:boolean), do: "bool"
  defp type_expr(:any), do: "typing.Any"
  defp type_expr({:list, type}), do: "list[#{type_expr(type)}]"

  defp type_expr({:dict, key_type, value_type}) do
    "dict[#{type_expr(key_type)}, #{type_expr(value_type)}]"
  end

  defp type_expr({:optional, type}), do: "typing.Optional[#{type_expr(type)}]"

  defp type_expr({:literal, values}) do
    "typing.Literal[" <> Enum.map_join(values, ", ", &literal_expr/1) <> "]"
  end

  defp literal_expr(value) when is_binary(value), do: Kernel.inspect(value)

  defp literal_expr(value) when is_atom(value) and value not in [nil, true, false] do
    Kernel.inspect(Atom.to_string(value))
  end

  defp literal_expr(true), do: "True"
  defp literal_expr(false), do: "False"
  defp literal_expr(value) when is_integer(value) or is_float(value), do: Kernel.to_string(value)
end
