defmodule Dsex.ModulesTest do
  use ExUnit.Case, async: true

  describe "Dsex.Predict" do
    test "answers from the configured LM" do
      lm = Dsex.LM.dummy([%{answer: "Paris"}])
      program = Dsex.Predict.new("question -> answer")

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(program, question: "What is the capital of France?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "Dsex.ChainOfThought" do
    test "exposes reasoning alongside the declared outputs" do
      lm = Dsex.LM.dummy([%{plan: "Because Paris is the capital.", answer: "Paris"}])
      program = Dsex.ChainOfThought.new("question -> answer")

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(program, question: "What is the capital of France?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
      assert Dsex.Prediction.get(prediction, :reasoning) =~ "Paris"
    end
  end

  describe "Dsex.ReAct" do
    test "calls an Elixir tool and finishes with the submitted answer" do
      weather = fn %{city: city} -> "sunny in " <> city end

      lm =
        Dsex.LM.dummy([
          %{
            next_thought: "I should check the weather.",
            tool_calls: %{tool_calls: [%{name: "weather", args: %{city: "Paris"}}]}
          },
          %{
            next_thought: "I can answer now.",
            tool_calls: %{tool_calls: [%{name: "submit", args: %{answer: "sunny in Paris"}}]}
          }
        ])

      agent =
        Dsex.ReAct.new("question -> answer",
          tools: [{"weather", "Look up the current weather for a city.", weather}]
        )

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(agent, question: "What's the weather like in Paris?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "sunny in Paris"
    end
  end

  describe "Dsex.BestOfN" do
    test "returns the reward_fn-selected prediction" do
      lm = Dsex.LM.dummy([%{plan: "Known fact.", answer: "Paris"}])
      qa = Dsex.ChainOfThought.new("question -> answer")

      reward = fn _inputs, pred ->
        if Dsex.Prediction.get(pred, :answer) == "Paris", do: 1.0, else: 0.0
      end

      best = Dsex.BestOfN.new(qa, n: 1, reward_fn: reward, threshold: 1.0)

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(best, question: "What is the capital of France?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "Dsex.Refine" do
    test "returns the reward_fn-selected prediction" do
      lm = Dsex.LM.dummy([%{plan: "Known fact.", answer: "Paris"}])
      qa = Dsex.ChainOfThought.new("question -> answer")

      reward = fn _inputs, pred ->
        if Dsex.Prediction.get(pred, :answer) == "Paris", do: 1.0, else: 0.0
      end

      refined = Dsex.Refine.new(qa, n: 1, reward_fn: reward, threshold: 1.0)

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(refined, question: "What is the capital of France?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "Dsex.MultiChainComparison" do
    test "picks a final answer given M attempts" do
      lm = Dsex.LM.dummy([%{rationale: "Holistic pick.", answer: "Paris"}])
      program = Dsex.MultiChainComparison.new("question -> answer", m: 2)

      completions = [
        %{question: "capital of France?", answer: "Paris", rationale: "r1"},
        %{question: "capital of France?", answer: "Paris", rationale: "r2"}
      ]

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(program, question: "capital of France?", completions: completions)
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "Dsex.Parallel" do
    test "runs (program, inputs) pairs" do
      lm = Dsex.LM.dummy([%{answer: "4"}, %{answer: "6"}])
      qa = Dsex.Predict.new("question -> answer")
      pairs = [{qa, %{question: "2+2?"}}, {qa, %{question: "3+3?"}}]

      results =
        Dsex.context([lm: lm], fn ->
          Dsex.Parallel.run(pairs, num_threads: 1)
        end)

      assert Enum.map(results, &Dsex.Prediction.get(&1, :answer)) == ["4", "6"]
    end
  end
end
