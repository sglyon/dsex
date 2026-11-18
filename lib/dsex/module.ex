defmodule Dsex.Module do
  @moduledoc """
  `use Dsex.Module` turns an Elixir module into a DSPy program whose
  predictors are declared with `predictor/2,3` and whose control flow is an
  Elixir `forward/2` callback.

  ## Example

      defmodule RAG do
        use Dsex.Module

        predictor :generate, Dsex.ChainOfThought, "context, question -> answer"

        def forward(prog, %{question: question}) do
          context = search(question)
          Dsex.Program.call(prog.generate, context: context, question: question)
        end
      end

      program = RAG.new()
      Dsex.Program.call(program, question: "...")

  Declared sub-programs are real attributes on the underlying `dspy.Module`:
  `Dsex.Program.named_predictors/1`, `deepcopy/1`, `save/2`, and optimizers
  that mutate copies all see them. `forward/2` always receives a fresh
  accessor (a map of `name => %Dsex.Program{}`) built from whichever
  instance DSPy is currently calling — essential when an optimizer calls a
  deep copy, since each copy's predictors must be independently mutable.

  A sub-program may itself be another `use Dsex.Module` program: declare it
  with `predictor name, OtherModule` (no signature — `OtherModule.new/0`).
  """

  @doc """
  Declares a named sub-program.

  `builder` is a module with a `new/0`, `new/1`, or `new/2` function
  returning a `%Dsex.Program{}` — a built-in (`Dsex.Predict`,
  `Dsex.ChainOfThought`, ...), or another `use Dsex.Module` program.

  `signature` is forwarded to `builder.new/1` as-is (a signature spec, e.g.
  a string or `Dsex.Signature` module), or as `{signature, opts}` to call
  `builder.new/2`. Omit it (or pass `nil`) to call `builder.new/0`, for
  nesting another custom module.

  ## Examples

      predictor :generate, Dsex.ChainOfThought, "context, question -> answer"
      predictor :retrieve, Dsex.Predict, {"question -> query", [temperature: 0.0]}
      predictor :sub, OtherRAGModule
  """
  defmacro predictor(name, builder, signature \\ nil) do
    quote do
      @dsex_predictors {unquote(name), unquote(builder), unquote(signature)}
    end
  end

  defmacro __using__(_opts) do
    quote do
      import Dsex.Module, only: [predictor: 2, predictor: 3]
      Module.register_attribute(__MODULE__, :dsex_predictors, accumulate: true)
      @before_compile Dsex.Module
    end
  end

  defmacro __before_compile__(env) do
    predictors = env.module |> Module.get_attribute(:dsex_predictors) |> Enum.reverse()

    quote do
      @doc """
      Builds a fresh `%Dsex.Program{}` backed by a Python `ElixirModule`,
      with its own copies of the declared sub-programs.
      """
      @spec new() :: Dsex.Program.t()
      def new do
        Dsex.Module.__build__(unquote(Macro.escape(predictors)), &__MODULE__.forward/2)
      end
    end
  end

  @doc false
  @spec __build__([{atom(), module(), term()}], (map(), map() -> term())) :: Dsex.Program.t()
  def __build__(predictors, forward_fun) do
    names = Enum.map(predictors, fn {name, _builder, _signature} -> name end)

    sub_programs =
      Map.new(predictors, fn {name, builder, signature} ->
        {Atom.to_string(name), build_sub(builder, signature)}
      end)

    wrapper = fn program, kwargs ->
      accessor = accessor(program, names)
      result = forward_fun.(accessor, Dsex.Convert.atomize_keys(kwargs))
      to_prediction(result)
    end

    elixir_fn = Dsex.Bridge.to_python(wrapper, name: "forward")

    Dsex.Python.call(
      "dsex_module.new_module",
      [],
      predictors: sub_programs,
      forward_fn: elixir_fn
    )
  end

  defp build_sub(builder, nil), do: builder.new()
  defp build_sub(builder, {signature, opts}), do: builder.new(signature, opts)
  defp build_sub(builder, signature), do: builder.new(signature)

  defp accessor(program, names) do
    Map.new(names, fn name -> {name, Dsex.Python.getattr(program, Atom.to_string(name))} end)
  end

  defp to_prediction(%Dsex.Prediction{} = prediction), do: prediction

  defp to_prediction(result) when is_list(result) do
    if Keyword.keyword?(result), do: result |> Map.new() |> to_prediction(), else: result
  end

  defp to_prediction(result) when is_map(result) do
    Dsex.Python.call("dspy.Prediction", [], result)
  end
end
