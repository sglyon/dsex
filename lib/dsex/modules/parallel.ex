defmodule Dsex.Parallel do
  @moduledoc """
  Wraps `dspy.Parallel`: runs `(program, inputs)` pairs across threads.
  """

  @doc """
  Runs each `{program, inputs}` pair in `pairs` (`inputs` a map/keyword list
  or `%Dsex.Example{}`) via `dspy.Parallel`. `opts`: `num_threads:`,
  `max_errors:`, `return_failed_examples:`, `disable_progress_bar:`,
  `timeout:`, `straggler_limit:`.

  ## Examples

      iex> qa = Dsex.Predict.new("question -> answer")
      iex> pairs = [{qa, %{question: "2+2?"}}, {qa, %{question: "3+3?"}}]
      iex> lm = Dsex.LM.dummy([%{answer: "4"}, %{answer: "6"}])
      iex> results = Dsex.context([lm: lm], fn -> Dsex.Parallel.run(pairs, num_threads: 1) end)
      iex> Enum.map(results, &Dsex.Prediction.get(&1, :answer))
      ["4", "6"]
  """
  @spec run([{Dsex.Program.t(), map() | keyword() | Dsex.Example.t()}], keyword()) :: [
          Dsex.Prediction.t()
        ]
  def run(pairs, opts \\ []) do
    normalized = Enum.map(pairs, &normalize_pair/1)
    parallel = Dsex.Python.call("dspy.Parallel", [], opts)
    Dsex.Python.call_method(parallel, "__call__", [normalized])
  end

  defp normalize_pair({program, %Dsex.Example{} = example}), do: {program, example}
  defp normalize_pair({program, inputs}), do: {program, Map.new(inputs)}
end
