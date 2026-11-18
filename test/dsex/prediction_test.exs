defmodule Dsex.PredictionTest do
  use ExUnit.Case, async: true

  test "new/1, get/3 and to_map/1 round-trip plain fields" do
    prediction = Dsex.Prediction.new(%{answer: "4", confidence: 0.9})

    assert Dsex.Prediction.get(prediction, :answer) == "4"
    assert Dsex.Prediction.get(prediction, :missing, "fallback") == "fallback"
    assert Dsex.Prediction.to_map(prediction) == %{answer: "4", confidence: 0.9}
  end

  test "completions/1 falls back to to_map/1 when none were recorded" do
    prediction = Dsex.Prediction.new(answer: "4")

    assert Dsex.Prediction.completions(prediction) == [%{answer: "4"}]
  end

  test "completions/1 returns one map per candidate when n > 1" do
    lm = Dsex.LM.dummy([%{answer: "red"}, %{answer: "blue"}, %{answer: "green"}])

    prediction =
      Dsex.context([lm: lm], fn ->
        predictor = Dsex.Python.call("dspy.Predict", ["question -> answer"], n: 3)
        Dsex.Python.call(predictor, [], question: "What color is the sky?")
      end)

    completions = Dsex.Prediction.completions(prediction)

    assert length(completions) == 3
    assert Enum.map(completions, & &1.answer) == ["red", "blue", "green"]
  end
end
