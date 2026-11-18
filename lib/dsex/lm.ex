defmodule Dsex.LM do
  @moduledoc """
  Wraps `dspy.LM` (a real backend) and `dspy.utils.DummyLM` (a deterministic,
  offline stand-in used in tests). Pair with `Dsex.configure/1` or
  `Dsex.context/2` to make DSPy programs use it.
  """

  use Dsex.Wrapper

  @history_keys ~w(prompt messages kwargs outputs usage cost timestamp uuid model model_type)

  @doc """
  Builds a `dspy.LM` for `model` (e.g. `"openai/gpt-4o-mini"`). `opts` are
  passed through as keyword arguments (`temperature`, `max_tokens`, `cache`,
  `api_key`, `api_base`, ...).

  ## Examples

      iex> Dsex.LM.new("openai/gpt-4o-mini", temperature: 0.0)
  """
  @spec new(String.t(), keyword()) :: t()
  def new(model, opts \\ []) when is_binary(model) do
    Dsex.Python.call("dspy.LM", [model], opts)
  end

  @doc """
  Builds a `dspy.utils.DummyLM` for offline, deterministic tests.

  `answers` is either a list of maps (one per call, returned in order) or a
  map of prompt substring to answer map (matched against the final user
  message). `opts` supports `follow_examples:`, `reasoning:` and `adapter:`.

  ## Examples

      iex> lm = Dsex.LM.dummy([%{answer: "4"}])
      iex> Dsex.context([lm: lm], fn ->
      ...>   Dsex.Python.call("dspy.Predict", ["question -> answer"])
      ...> end)
  """
  @spec dummy([map()] | map(), keyword()) :: t()
  def dummy(answers, opts \\ []) when is_list(answers) or is_map(answers) do
    Dsex.Python.call("dspy.utils.DummyLM", [answers], opts)
  end

  @doc """
  Returns `lm`'s call history as a list of maps (atom keys, including
  nested maps), oldest first.

  Only serializable fields are kept (`prompt`, `messages`, `kwargs`,
  `outputs`, `usage`, `cost`, `timestamp`, `uuid`, `model`, `model_type`);
  raw provider response objects (`request`, `response`, `response_model`)
  are dropped since they do not decode cleanly.

  ## Examples

      iex> lm = Dsex.LM.dummy([%{answer: "4"}])
      iex> Dsex.context([lm: lm], fn ->
      ...>   Dsex.Python.call("dspy.Predict", ["question -> answer"], question: "2+2?")
      ...> end)
      iex> [%{outputs: _} | _] = Dsex.LM.history(lm)
  """
  @spec history(t()) :: [map()]
  def history(%__MODULE__{} = lm) do
    code = """
    keys = #{inspect(@history_keys)}
    [{k: entry[k] for k in keys if k in entry} for entry in lm.history]
    """

    code |> Dsex.Python.eval(%{"lm" => lm}) |> Enum.map(&atomize/1)
  end

  @doc """
  Captures `dspy.inspect_history/2`'s pretty-printed output (the last `n`
  calls of DSPy's global history, shared across every LM) as a plain
  string, with ANSI color codes disabled.

  ## Examples

      iex> lm = Dsex.LM.dummy([%{answer: "4"}])
      iex> Dsex.context([lm: lm], fn ->
      ...>   Dsex.Python.call("dspy.Predict", ["question -> answer"], question: "2+2?")
      ...> end)
      iex> Dsex.LM.inspect_history(1) =~ "2+2?"
      true
  """
  @spec inspect_history(pos_integer()) :: String.t()
  def inspect_history(n \\ 1) when is_integer(n) and n > 0 do
    code = """
    import io

    import dspy

    buf = io.StringIO()
    dspy.inspect_history(n, file=buf)
    buf.getvalue()
    """

    Dsex.Python.eval(code, %{"n" => n})
  end

  @doc """
  Returns summed token usage across `lm`'s history, as a map with atom keys
  (e.g. `%{prompt_tokens: 10, completion_tokens: 4, total_tokens: 14}`), or
  an empty map when no usage was recorded.

  ## Examples

      iex> Dsex.LM.usage(Dsex.LM.dummy([%{answer: "4"}]))
      %{}
  """
  @spec usage(t()) :: map()
  def usage(%__MODULE__{} = lm) do
    lm
    |> history()
    |> Enum.reduce(%{}, fn entry, acc ->
      entry |> Map.get(:usage, %{}) |> merge_usage(acc)
    end)
  end

  defp merge_usage(usage, acc) when is_map(usage) do
    Enum.reduce(usage, acc, fn
      {key, value}, acc when is_number(value) -> Map.update(acc, key, value, &(&1 + value))
      _pair, acc -> acc
    end)
  end

  defp merge_usage(_usage, acc), do: acc

  defp atomize(term) when is_map(term) do
    term |> Dsex.Convert.atomize_keys() |> Map.new(fn {k, v} -> {k, atomize(v)} end)
  end

  defp atomize(term) when is_list(term), do: Enum.map(term, &atomize/1)
  defp atomize(term), do: term
end
