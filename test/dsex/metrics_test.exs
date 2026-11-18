defmodule Dsex.MetricsTest do
  use ExUnit.Case, async: true

  describe "normalize_text/1" do
    test "lowercases, drops articles/punctuation, collapses whitespace" do
      assert Dsex.Metrics.normalize_text("The Eiffel Tower!") == "eiffel tower"
    end
  end

  describe "exact_match/2" do
    test "matches after normalization" do
      assert Dsex.Metrics.exact_match("The Eiffel Tower", "eiffel tower")
      refute Dsex.Metrics.exact_match("The Louvre", "eiffel tower")
    end
  end

  describe "f1_score/2" do
    test "computes token-level F1 between normalized strings" do
      assert_in_delta Dsex.Metrics.f1_score("eiffel tower paris", "eiffel tower"), 0.8, 0.001
      assert Dsex.Metrics.f1_score("eiffel tower", "eiffel tower") == 1.0
    end
  end

  describe "answer_exact_match/0,1" do
    test "requires an exact match after normalization by default" do
      example = Dsex.Example.new(%{question: "Q", answer: "Eiffel Tower"}, inputs: [:question])
      pred = Dsex.Prediction.new(%{answer: "Eiffel"})

      refute Dsex.Python.call(Dsex.Metrics.answer_exact_match(), [example, pred])
    end

    test "accepts a sufficiently high partial F1 match when given frac" do
      example = Dsex.Example.new(%{question: "Q", answer: "Eiffel Tower"}, inputs: [:question])
      pred = Dsex.Prediction.new(%{answer: "Eiffel"})

      assert Dsex.Python.call(Dsex.Metrics.answer_exact_match(0.5), [example, pred])
      refute Dsex.Python.call(Dsex.Metrics.answer_exact_match(0.9), [example, pred])
    end

    test "is usable directly as a Dsex.Evaluate metric" do
      example =
        Dsex.Example.new(%{question: "capital of France?", answer: "Paris"}, inputs: [:question])

      lm = Dsex.LM.dummy([%{answer: "Paris"}])
      program = Dsex.Python.call("dspy.Predict", ["question -> answer"])

      result =
        Dsex.context([lm: lm], fn ->
          Dsex.Evaluate.run(program, [example], Dsex.Metrics.answer_exact_match(), num_threads: 1)
        end)

      assert result.score == 100.0
    end
  end

  describe "answer_passage_match/0" do
    test "matches when a passage contains the reference answer" do
      example = Dsex.Example.new(%{question: "Q", answer: "Eiffel Tower"}, inputs: [:question])
      hit = Dsex.Prediction.new(%{context: ["The Eiffel Tower is in Paris."]})
      miss = Dsex.Prediction.new(%{context: ["The Louvre is in Paris."]})

      assert Dsex.Python.call(Dsex.Metrics.answer_passage_match(), [example, hit])
      refute Dsex.Python.call(Dsex.Metrics.answer_passage_match(), [example, miss])
    end
  end

  describe "semantic_f1/1 and complete_and_grounded/1" do
    test "construct LM-judged metric programs" do
      assert %Dsex.Program{} = Dsex.Metrics.semantic_f1(threshold: 0.5)
      assert %Dsex.Program{} = Dsex.Metrics.complete_and_grounded()
    end
  end
end
