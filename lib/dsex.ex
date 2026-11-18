defmodule Dsex do
  @moduledoc """
  DSPy for Elixir, backed by the real DSPy running in an embedded Python
  interpreter (`Pythonx`).
  """

  @doc """
  Sets process-wide DSPy defaults (`dspy.configure`), e.g. `lm:` and `adapter:`.
  Values are encoded with `Pythonx.Encoder`.
  """
  @spec configure(keyword()) :: :ok
  def configure(opts) when is_list(opts) do
    Dsex.Python.call("dsex_bridge.configure", [], opts)
    :ok
  end

  @doc """
  Runs `program` on `inputs` (map or keyword list). See `Dsex.Program.call/2`.
  """
  @spec call(Dsex.Program.t(), map() | keyword()) :: Dsex.Prediction.t()
  defdelegate call(program, inputs), to: Dsex.Program

  @doc """
  Runs `fun` with DSPy settings overridden (`dspy.context`), for every Dsex call
  made by the current process inside `fun`. Nested blocks merge, inner wins.
  """
  @spec context(keyword(), (-> result)) :: result when result: term()
  def context(opts, fun) when is_list(opts) and is_function(fun, 0) do
    stack = Process.get(:dsex_context, [])
    Process.put(:dsex_context, [opts | stack])

    try do
      fun.()
    after
      Process.put(:dsex_context, stack)
    end
  end
end
