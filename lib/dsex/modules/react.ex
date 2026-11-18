defmodule Dsex.ReAct do
  @moduledoc """
  Wraps `dspy.ReAct`: an agent that interleaves reasoning and tool calls
  until it submits the signature's declared outputs.
  """

  @doc """
  Builds a ReAct agent for `signature`.

  Options:

    * `:tools` — a list of `Dsex.Tool.spec/0` entries: bare Elixir functions
      (named `"tool_<n>"`), `{name, doc, fun}` tuples, or `%Dsex.Tool{}`
      structs built with `Dsex.Tool.new/2`.
    * `:max_iters` — maximum reasoning/tool-call turns (default `20`).

  ## Examples

      iex> weather = fn %{city: city} -> "sunny in " <> city end
      iex> agent = Dsex.ReAct.new("question -> answer",
      ...>   tools: [{"weather", "Look up the weather for a city.", weather}]
      ...> )
      iex> match?(%Dsex.Program{}, agent)
      true
  """
  @spec new(term(), keyword()) :: Dsex.Program.t()
  def new(signature, opts \\ []) do
    sig = Dsex.Signature.from(signature)
    {tools, opts} = Keyword.pop(opts, :tools, [])

    py_tools =
      tools
      |> Enum.with_index()
      |> Enum.map(fn {tool, index} -> Dsex.Tool.coerce(tool, index) end)

    Dsex.Python.call("dspy.ReAct", [sig, py_tools], opts)
  end
end
