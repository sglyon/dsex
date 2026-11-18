defmodule Dsex.Wrapper do
  @moduledoc false
  # `use Dsex.Wrapper` defines a struct with a single `:py` field holding the
  # underlying `Pythonx.Object`, plus `Pythonx.Encoder` (pass the object
  # through) and `Inspect` (show the Python repr) implementations.

  defmacro __using__(_opts) do
    quote do
      @enforce_keys [:py]
      defstruct [:py]

      @type t :: %__MODULE__{py: Pythonx.Object.t()}

      defimpl Pythonx.Encoder do
        def encode(%{py: py}, _encoder), do: py
      end

      defimpl Inspect do
        def inspect(%{py: py}, _opts) do
          repr = Dsex.Python.call("builtins.repr", [py])
          "#" <> Kernel.inspect(@for) <> "<" <> repr <> ">"
        end
      end
    end
  end
end
