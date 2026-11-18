defmodule Dsex.Python do
  @moduledoc """
  Low-level access to the embedded Python interpreter running DSPy.

  Dsex embeds CPython in the BEAM through `Pythonx`. On first use,
  `ensure_initialized!/0` installs a uv-managed environment containing DSPy
  (configurable via `config :dsex, :python`) and loads the `dsex_bridge`
  helper module from `priv/python/dsex_bridge.py`.

  Every call made through this module:

    * encodes arguments with `Pythonx.Encoder` (Elixir functions become
      Python callables backed by `Dsex.Bridge`, wrapper structs such as
      `Dsex.Example` pass their underlying Python object);
    * runs under the active `dspy.context` overrides (see `Dsex.context/2`
      and callbacks from Python, which carry the caller's overrides);
    * decodes the result with `Dsex.Convert.from_py/1`, which wraps DSPy
      objects into Dsex structs and leaves other objects as `Pythonx.Object`.

  ## Configuration

      config :dsex, :python,
        python: "==3.13.*",
        dependencies: ["dspy[optuna]==3.5.0"],
        extra_dependencies: []

  If Pythonx was already initialized (for example by `config :pythonx, :uv_init`
  for releases), Dsex reuses that interpreter; it must include `dspy`.
  """

  @default_python "==3.13.*"
  @default_dependencies ["dspy[optuna]==3.5.0"]
  @ready_key {__MODULE__, :ready}

  @doc "Returns the pyproject.toml Dsex uses for `Pythonx.uv_init/2`."
  @spec pyproject(keyword()) :: String.t()
  def pyproject(opts \\ []) do
    opts = Keyword.merge(Application.get_env(:dsex, :python, []), opts)
    python = Keyword.get(opts, :python, @default_python)

    deps =
      Keyword.get(opts, :dependencies, @default_dependencies) ++
        Keyword.get(opts, :extra_dependencies, [])

    """
    [project]
    name = "dsex_python_env"
    version = "0.0.0"
    requires-python = "#{python}"
    dependencies = [#{Enum.map_join(deps, ", ", &inspect/1)}]
    """
  end

  @doc """
  Initializes Python (if needed) and loads the bridge. Idempotent and safe to
  call concurrently.
  """
  @spec ensure_initialized!(keyword()) :: :ok
  def ensure_initialized!(opts \\ []) do
    if :persistent_term.get(@ready_key, false) do
      :ok
    else
      :global.trans({__MODULE__, self()}, fn -> do_init(opts) end, [node()])
    end
  end

  defp do_init(opts) do
    if :persistent_term.get(@ready_key, false) do
      :ok
    else
      unless pythonx_initialized?() do
        Pythonx.uv_init(pyproject(opts))
      end

      bridge_pid = Process.whereis(Dsex.Bridge) || raise "Dsex application is not started"
      dir = Path.join(:code.priv_dir(:dsex), "python")
      others = dir |> File.ls!() |> Enum.filter(&String.ends_with?(&1, ".py")) |> Enum.sort()

      # dsex_bridge.py first; the other files import it.
      for file <- Enum.uniq(["dsex_bridge.py" | others]) do
        Pythonx.eval(File.read!(Path.join(dir, file)), %{"bridge_pid" => bridge_pid})
      end

      :persistent_term.put(@ready_key, true)
      :ok
    end
  end

  defp pythonx_initialized?, do: :persistent_term.get(:pythonx_init_state, nil) != nil

  @doc """
  Returns true when `a` and `b` reference the same Python object (`a is b`).
  Two `Pythonx.Object`s (or wrapper structs) for one Python object are not
  `==` in Elixir, so use this to compare identity.
  """
  @spec same?(Pythonx.Object.t() | struct(), Pythonx.Object.t() | struct()) :: boolean()
  def same?(a, b), do: call("operator.is_", [a, b])

  @doc """
  Calls a Python callable and returns the decoded result.

  `target` is a dotted path (`"dspy.Predict"`), a `Pythonx.Object`, or a Dsex
  wrapper struct. `kwargs` is a map or keyword list.
  """
  @spec call(String.t() | Pythonx.Object.t() | struct(), list(), map() | keyword()) :: term()
  def call(target, args \\ [], kwargs \\ %{}) do
    invoke(target, nil, args, kwargs)
  end

  @doc "Calls method `name` on `object` (a `Pythonx.Object` or wrapper struct)."
  @spec call_method(Pythonx.Object.t() | struct(), String.t(), list(), map() | keyword()) ::
          term()
  def call_method(object, name, args \\ [], kwargs \\ %{}) when is_binary(name) do
    invoke(object, name, args, kwargs)
  end

  @doc "Reads attribute `name` of `object` and decodes it."
  @spec getattr(Pythonx.Object.t() | struct(), String.t()) :: term()
  def getattr(object, name) when is_binary(name) do
    call("builtins.getattr", [object, name])
  end

  @doc "Sets attribute `name` of `object` to `value`."
  @spec setattr(Pythonx.Object.t() | struct(), String.t(), term()) :: :ok
  def setattr(object, name, value) when is_binary(name) do
    call("builtins.setattr", [object, name, value])
    :ok
  end

  @doc """
  Evaluates raw Python `code` with `globals` (encoded with
  `Dsex.Convert.to_py/1`) and returns the decoded value of the final
  expression; DSPy objects in the result stay `Pythonx.Object`s. Runs outside
  `dspy.context` overrides; prefer `call/3` for DSPy calls.
  """
  @spec eval(String.t(), map()) :: term()
  def eval(code, globals \\ %{}) do
    ensure_initialized!()
    globals = Map.new(globals, fn {k, v} -> {to_string(k), Dsex.Convert.to_py(v)} end)

    case Pythonx.eval("import dsex_bridge\n" <> code, globals) do
      {nil, _} -> nil
      {result, _} -> Dsex.Convert.from_py(result)
    end
  end

  defp invoke(target, method, args, kwargs) do
    ensure_initialized!()

    {result, _globals} =
      Pythonx.eval(
        """
        import dsex_bridge
        dsex_bridge.invoke(base, ctx, target if method is None else getattr(target, method), args, kwargs)
        """,
        %{
          "base" => Process.get(:dsex_py_overrides),
          "ctx" => Dsex.Convert.to_py_kwargs(context_overrides()),
          "target" => Dsex.Convert.to_py(target),
          "method" => Dsex.Convert.to_py(method),
          "args" => Dsex.Convert.to_py(args),
          "kwargs" => Dsex.Convert.to_py_kwargs(kwargs)
        }
      )

    Dsex.Convert.from_py(result)
  end

  @doc false
  # Merged keyword overrides from enclosing Dsex.context/2 blocks.
  def context_overrides do
    Process.get(:dsex_context, []) |> Enum.reverse() |> Enum.reduce([], &Keyword.merge(&2, &1))
  end
end
