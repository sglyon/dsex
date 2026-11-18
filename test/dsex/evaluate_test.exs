defmodule Dsex.EvaluateTest do
  use ExUnit.Case, async: true

  setup do
    program = Dsex.Python.call("dspy.Predict", ["question -> answer"])
    %{program: program}
  end

  test "scores with a 2-arity Elixir metric function", %{program: program} do
    devset = [
      Dsex.Example.new(%{question: "capital of France?", answer: "Paris"}, inputs: [:question]),
      Dsex.Example.new(%{question: "capital of England?", answer: "Berlin"}, inputs: [:question])
    ]

    lm = Dsex.LM.dummy([%{answer: "Paris"}, %{answer: "London"}])

    metric = fn example, pred ->
      Dsex.Example.get(example, :answer) == Dsex.Prediction.get(pred, :answer)
    end

    result =
      Dsex.context([lm: lm], fn ->
        Dsex.Evaluate.run(program, devset, metric, num_threads: 1)
      end)

    assert result.score == 50.0
    assert [{_ex1, _pred1, score1}, {_ex2, _pred2, score2}] = result.results
    assert [score1, score2] == [true, false]
  end

  test "scores with Dsex.Metrics.answer_exact_match/0", %{program: program} do
    devset = [
      Dsex.Example.new(%{question: "capital of France?", answer: "Paris"}, inputs: [:question])
    ]

    lm = Dsex.LM.dummy([%{answer: "Paris"}])

    result =
      Dsex.context([lm: lm], fn ->
        Dsex.Evaluate.run(program, devset, Dsex.Metrics.answer_exact_match(),
          num_threads: 1,
          display_progress: false
        )
      end)

    assert result.score == 100.0
    assert [{example, prediction, true}] = result.results
    assert Dsex.Example.get(example, :question) == "capital of France?"
    assert Dsex.Prediction.get(prediction, :answer) == "Paris"
  end
end
