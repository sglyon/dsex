defmodule Dsex.Metrics do
  @moduledoc """
  DSPy's built-in evaluation metrics, usable anywhere a metric is accepted:
  `Dsex.Evaluate.run/4`, any `Dsex.Optimizer`, or a plain `Dsex.Program.call/2`
  comparison in your own Elixir metric.

  Functions that return a metric (`answer_exact_match/0,1`,
  `answer_passage_match/0`, `semantic_f1/1`, `complete_and_grounded/1`) hand
  back the underlying Python callable (a `%Pythonx.Object{}`, or a
  `%Dsex.Program{}` for the LM-judged metrics, which also encodes back to a
  plain callable). The remaining functions are pure helpers that call into
  `dspy.evaluate.metrics` directly and return a decoded Elixir value.
  """

  alias Dsex.Python

  @doc """
  The exact-match (or F1-thresholded, via `frac`) metric: compares
  `pred.answer` against `example.answer` (a string or list of reference
  strings).

  ## Examples

      iex> Dsex.Metrics.answer_exact_match()
      #Pythonx.Object<...>
  """
  @spec answer_exact_match() :: Pythonx.Object.t()
  def answer_exact_match do
    Python.call("dsex_bridge.resolve", ["dspy.evaluate.metrics.answer_exact_match"])
  end

  @doc """
  `answer_exact_match/0`, but accepting a match as soon as the per-reference
  F1 score reaches `frac` instead of requiring an exact match.

  ## Examples

      iex> Dsex.Metrics.answer_exact_match(0.5)
      #Pythonx.Object<...>
  """
  @spec answer_exact_match(number()) :: Pythonx.Object.t()
  def answer_exact_match(frac) when is_number(frac) do
    Python.call("functools.partial", [answer_exact_match()], %{"frac" => frac})
  end

  @doc """
  Metric that succeeds when any passage in `pred.context` contains one of the
  reference answers in `example.answer`.

  ## Examples

      iex> Dsex.Metrics.answer_passage_match()
      #Pythonx.Object<...>
  """
  @spec answer_passage_match() :: Pythonx.Object.t()
  def answer_passage_match do
    Python.call("dsex_bridge.resolve", ["dspy.evaluate.metrics.answer_passage_match"])
  end

  @doc """
  LM-judged semantic F1 between `pred.response` and `example.response`, via
  `dspy.evaluate.auto_evaluation.SemanticF1`. `opts`: `:threshold` (default
  `0.66`), `:decompositional` (default `false`).

  Needs an LM configured (`Dsex.configure/1` or `Dsex.context/2`) to run.

  ## Examples

      iex> Dsex.Metrics.semantic_f1(threshold: 0.8)
      %Dsex.Program{}
  """
  @spec semantic_f1(keyword() | map()) :: Dsex.Program.t()
  def semantic_f1(opts \\ []) do
    Python.call("dspy.evaluate.auto_evaluation.SemanticF1", [], opts)
  end

  @doc """
  LM-judged combination of answer completeness and groundedness, via
  `dspy.evaluate.auto_evaluation.CompleteAndGrounded`. `opts`: `:threshold`
  (default `0.66`).

  Needs an LM configured (`Dsex.configure/1` or `Dsex.context/2`) to run.

  ## Examples

      iex> Dsex.Metrics.complete_and_grounded(threshold: 0.8)
      %Dsex.Program{}
  """
  @spec complete_and_grounded(keyword() | map()) :: Dsex.Program.t()
  def complete_and_grounded(opts \\ []) do
    Python.call("dspy.evaluate.auto_evaluation.CompleteAndGrounded", [], opts)
  end

  @doc """
  Normalizes `text` the way DSPy's metrics do before comparing: lowercases,
  drops English articles and punctuation, collapses whitespace.

  ## Examples

      iex> Dsex.Metrics.normalize_text("The Eiffel Tower!")
      "eiffel tower"
  """
  @spec normalize_text(String.t()) :: String.t()
  def normalize_text(text) when is_binary(text) do
    Python.call("dspy.evaluate.metrics.normalize_text", [text])
  end

  @doc """
  Boolean exact match between `prediction` and `ground_truth` after
  `normalize_text/1`.

  ## Examples

      iex> Dsex.Metrics.exact_match("The Eiffel Tower", "eiffel tower")
      true
  """
  @spec exact_match(String.t(), String.t()) :: boolean()
  def exact_match(prediction, ground_truth)
      when is_binary(prediction) and is_binary(ground_truth) do
    Python.call("dspy.evaluate.metrics.em_score", [prediction, ground_truth])
  end

  @doc """
  Token-level F1 between `prediction` and `ground_truth` after
  `normalize_text/1`.

  ## Examples

      iex> Dsex.Metrics.f1_score("eiffel tower paris", "eiffel tower")
      0.8
  """
  @spec f1_score(String.t(), String.t()) :: float()
  def f1_score(prediction, ground_truth)
      when is_binary(prediction) and is_binary(ground_truth) do
    Python.call("dspy.evaluate.metrics.f1_score", [prediction, ground_truth])
  end
end
