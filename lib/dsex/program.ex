defmodule Dsex.Program do
  @moduledoc """
  Calls and introspects any `dspy.Module` instance: built-ins from
  `Dsex.Predict`, `Dsex.ChainOfThought`, `Dsex.ReAct`, etc., custom
  `use Dsex.Module` programs, and the result of optimizer `compile/2,3`.

  A `%Dsex.Program{}` only wraps a Python `dspy.Module` object; every
  function here delegates to that object through `Dsex.Python`.
  """

  use Dsex.Wrapper

  alias Dsex.Python

  @doc """
  Runs `program` on `inputs` (a map or keyword list of its input fields),
  i.e. `program(**inputs)` in DSPy terms.

  ## Examples

      iex> program = Dsex.Predict.new("question -> answer")
      iex> lm = Dsex.LM.dummy([%{answer: "Paris"}])
      iex> prediction = Dsex.context([lm: lm], fn ->
      ...>   Dsex.Program.call(program, question: "What is the capital of France?")
      ...> end)
      iex> Dsex.Prediction.get(prediction, :answer)
      "Paris"
  """
  @spec call(t(), map() | keyword()) :: Dsex.Prediction.t()
  def call(program, inputs) do
    Python.call(program, [], inputs)
  end

  @doc """
  Lists `{name, %Dsex.Program{}}` pairs for every `dspy.Predict` the program
  contains, named by their attribute path.
  """
  @spec named_predictors(t()) :: [{String.t(), t()}]
  def named_predictors(program) do
    Python.call_method(program, "named_predictors")
  end

  @doc "Lists the predictors themselves, in the order `named_predictors/1` returns them."
  @spec predictors(t()) :: [t()]
  def predictors(program) do
    program |> named_predictors() |> Enum.map(fn {_name, predictor} -> predictor end)
  end

  @doc "Deep-copies `program`, including every predictor's demos and LM config."
  @spec deepcopy(t()) :: t()
  def deepcopy(program), do: Python.call_method(program, "deepcopy")

  @doc "Deep-copies `program` and resets every predictor's learned state (demos, instructions)."
  @spec reset_copy(t()) :: t()
  def reset_copy(program), do: Python.call_method(program, "reset_copy")

  @doc "Sets the LM every predictor in `program` uses."
  @spec set_lm(t(), Dsex.LM.t()) :: :ok
  def set_lm(program, lm) do
    Python.call_method(program, "set_lm", [lm])
    :ok
  end

  @doc """
  Saves `program`'s state (demos, instructions, LM config) to `path`, as
  JSON or pickle depending on its extension (`.json` or `.pkl`).
  `save_program` is always `false`: a custom `use Dsex.Module` program's
  Elixir forward callback cannot be pickled, only its predictors' state.
  """
  @spec save(t(), Path.t()) :: :ok
  def save(program, path) do
    Python.call_method(program, "save", [path], save_program: false)
    :ok
  end

  @doc "Loads state saved by `save/2` into `program` in place and returns it."
  @spec load(t(), Path.t()) :: t()
  def load(program, path) do
    Python.call_method(program, "load", [path])
    program
  end

  @doc "Returns `program`'s state (demos, instructions, LM config) as a plain map."
  @spec dump_state(t()) :: map()
  def dump_state(program), do: Python.call_method(program, "dump_state")

  @doc "Restores state previously produced by `dump_state/1`."
  @spec load_state(t(), map()) :: :ok
  def load_state(program, state) do
    Python.call_method(program, "load_state", [state])
    :ok
  end

  @doc """
  Runs `program` over `inputs` (each a map/keyword list or `%Dsex.Example{}`)
  across threads via `dspy.Module.batch`. `opts`: `num_threads:`,
  `max_errors:`, `return_failed_examples:`, `disable_progress_bar:`,
  `timeout:`, `straggler_limit:`.

  ## Examples

      iex> program = Dsex.Predict.new("question -> answer")
      iex> lm = Dsex.LM.dummy([%{answer: "4"}, %{answer: "6"}])
      iex> results = Dsex.context([lm: lm], fn ->
      ...>   Dsex.Program.batch(program, [%{question: "2+2?"}, %{question: "3+3?"}])
      ...> end)
      iex> Enum.map(results, &Dsex.Prediction.get(&1, :answer))
      ["4", "6"]
  """
  @spec batch(t(), [map() | keyword() | Dsex.Example.t()], keyword()) :: [Dsex.Prediction.t()]
  def batch(program, inputs, opts \\ []) do
    examples = Enum.map(inputs, &to_example/1)
    Python.call_method(program, "batch", [examples], opts)
  end

  defp to_example(%Dsex.Example{} = example), do: example

  defp to_example(input) do
    fields = Map.new(input)
    Dsex.Example.new(fields, inputs: Map.keys(fields))
  end
end
