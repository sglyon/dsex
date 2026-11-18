defmodule Dsex.Optimizer do
  @moduledoc """
  Generic wrapper around any DSPy `Teleprompter` (prompt optimizer).

  Every Dsex optimizer — whether built through `new/2` directly or through a
  named constructor in `Dsex.Optimizers.*` — is a `%Dsex.Optimizer{}` holding
  the underlying Python instance, compiled with `compile/3`.
  """

  use Dsex.Wrapper

  alias Dsex.Python

  @doc """
  Constructs a DSPy optimizer from a dotted Python class path, e.g.
  `"dspy.LabeledFewShot"`. `opts` become the class's constructor keyword
  arguments; Elixir functions (metrics, teacher settings, ...) are passed
  through as callables automatically.

  Prefer the named constructors in `Dsex.Optimizers` when one exists; use this
  directly for optimizers without one (weight fine-tuning optimizers,
  `dspy.BootstrapFinetune`, `dspy.AvatarOptimizer`, ...).

  ## Examples

      iex> Dsex.Optimizer.new("dspy.LabeledFewShot", k: 4)
      %Dsex.Optimizer{}
  """
  @spec new(String.t(), keyword() | map()) :: t()
  def new(class_path, opts \\ []) when is_binary(class_path) do
    %__MODULE__{py: Python.call(class_path, [], opts)}
  end

  @doc """
  Compiles `student` with `optimizer`, returning the optimized program.

  `opts` become the optimizer's `compile` keyword arguments (e.g. `trainset:`,
  `valset:`, `teacher:`, plus whatever that optimizer's `compile` accepts).
  `student` is passed positionally, so optimizers whose `compile` takes
  something other than a single program first (e.g.
  `Dsex.Optimizers.Ensemble`, which takes a list of programs) work the same
  way.

  ## Examples

      iex> Dsex.Optimizer.compile(optimizer, student, trainset: trainset)
      %Dsex.Program{}
  """
  @spec compile(t(), term(), keyword() | map()) :: Dsex.Program.t()
  def compile(%__MODULE__{} = optimizer, student, opts \\ []) do
    Python.call_method(optimizer, "compile", [student], opts)
  end
end
