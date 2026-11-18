defmodule Dsex.Bridge do
  @moduledoc """
  Routes calls from Python back into Elixir.

  An Elixir function passed to Python becomes a `dsex_bridge.ElixirFn`. When
  Python calls it (possibly from a DSPy worker thread), the bridge receives a
  message, runs the function in a supervised task, and hands the result back
  to the waiting Python thread. Arguments arrive decoded by
  `Dsex.Convert.from_py/1`.

  ## Arity

  The function is applied to the positional arguments Python passed,
  truncated or padded with `nil` to the function's arity. DSPy calls metrics
  as `metric(example, prediction)` during evaluation and
  `metric(example, prediction, trace)` during bootstrapping, so both
  `fn ex, pred -> ... end` and `fn ex, pred, trace -> ... end` work. Keyword
  arguments, if any, are passed as a final map argument (string keys converted
  to atoms) when the function's arity leaves room for it.

  ## Context

  The task running the function inherits the Python caller's `dspy.context`
  overrides (trace, LM, adapter), so DSPy calls it makes are recorded in the
  same trace — required for optimizers to bootstrap demos through Elixir code.
  """

  use GenServer

  @table __MODULE__

  @doc false
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc """
  Registers `fun` and returns a Python `ElixirFn` object wrapping it.

  Options: `:name` (Python `__name__`, used e.g. as a ReAct tool name) and
  `:doc` (Python `__doc__`).
  """
  @spec to_python(function(), keyword()) :: Pythonx.Object.t()
  def to_python(fun, opts \\ []) when is_function(fun) do
    Dsex.Python.ensure_initialized!()
    fid = :ets.update_counter(@table, :next_id, 1, {:next_id, 0})
    :ets.insert(@table, {fid, fun})

    {object, _} =
      Pythonx.eval(
        "import dsex_bridge\ndsex_bridge.ElixirFn(fid, name, doc)",
        %{
          "fid" => fid,
          "name" => Dsex.Convert.to_py(opts[:name]),
          "doc" => Dsex.Convert.to_py(opts[:doc])
        }
      )

    object
  end

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
    {:ok, %{}}
  end

  @impl true
  def handle_info({:dsex_call, request}, state) do
    {fid, args, kwargs, overrides, reply} = Pythonx.decode(request)

    Task.Supervisor.start_child(Dsex.TaskSupervisor, fn ->
      run(fid, args, kwargs, overrides, reply)
    end)

    {:noreply, state}
  end

  def handle_info({:dsex_release, fid}, state) do
    :ets.delete(@table, Pythonx.decode(fid))
    {:noreply, state}
  end

  defp run(fid, args, kwargs, overrides, reply) do
    Process.put(:dsex_py_overrides, overrides)

    result =
      try do
        [{^fid, fun}] = :ets.lookup(@table, fid)
        value = apply_fun(fun, Dsex.Convert.from_py(args), Dsex.Convert.from_py(kwargs))
        {"ok", Dsex.Convert.to_py(value)}
      rescue
        exception -> {"error", Exception.format(:error, exception, __STACKTRACE__)}
      catch
        kind, reason -> {"error", Exception.format(kind, reason, __STACKTRACE__)}
      end

    Pythonx.eval("reply.put(result)", %{"reply" => reply, "result" => Dsex.Convert.to_py(result)})
  end

  defp apply_fun(fun, args, kwargs) do
    {:arity, arity} = Function.info(fun, :arity)
    nargs = length(args)

    args =
      cond do
        map_size(kwargs) > 0 and arity == nargs + 1 -> args ++ [Dsex.Convert.atomize_keys(kwargs)]
        arity <= nargs -> Enum.take(args, arity)
        true -> args ++ List.duplicate(nil, arity - nargs)
      end

    apply(fun, args)
  end
end
