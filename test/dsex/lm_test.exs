defmodule Dsex.LMTest do
  use ExUnit.Case, async: true

  test "dummy/2 works under Dsex.context and drives a Predict call" do
    lm = Dsex.LM.dummy([%{answer: "4"}])

    answer =
      Dsex.context([lm: lm], fn ->
        predictor = Dsex.Python.call("dspy.Predict", ["question -> answer"])
        prediction = Dsex.Python.call(predictor, [], question: "2+2?")
        Dsex.Prediction.get(prediction, :answer)
      end)

    assert answer == "4"
  end

  test "history/1 is non-empty after a call and keeps atom-keyed maps" do
    lm = Dsex.LM.dummy([%{answer: "4"}])

    Dsex.context([lm: lm], fn ->
      predictor = Dsex.Python.call("dspy.Predict", ["question -> answer"])
      Dsex.Python.call(predictor, [], question: "2+2?")
    end)

    assert [%{outputs: outputs, model: "dummy"} | _] = Dsex.LM.history(lm)
    assert outputs != []
  end

  test "usage/1 returns an empty map when no usage was recorded" do
    assert Dsex.LM.usage(Dsex.LM.dummy([%{answer: "4"}])) == %{}
  end
end

defmodule Dsex.LM.InspectHistoryTest do
  # dspy.inspect_history/2 reads DSPy's GLOBAL_HISTORY, shared across every
  # LM and every test process; async: false guarantees no other test runs
  # concurrently, so the entry we just recorded is guaranteed to be last.
  use ExUnit.Case, async: false

  test "inspect_history/1 returns a string containing the prompt" do
    lm = Dsex.LM.dummy([%{answer: "4"}])

    Dsex.context([lm: lm], fn ->
      predictor = Dsex.Python.call("dspy.Predict", ["question -> answer"])
      Dsex.Python.call(predictor, [], question: "What is 2+2?")
    end)

    history_text = Dsex.LM.inspect_history(1)

    assert is_binary(history_text)
    assert history_text =~ "What is 2+2?"
  end
end
