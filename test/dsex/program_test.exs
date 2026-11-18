defmodule Dsex.ProgramTest do
  use ExUnit.Case, async: true

  describe "call/2" do
    test "runs a Predict program and returns a Prediction" do
      lm = Dsex.LM.dummy([%{answer: "Paris"}])
      program = Dsex.Predict.new("question -> answer")

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(program, question: "What is the capital of France?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "named_predictors/1 and predictors/1" do
    test "lists the program's Predict instances" do
      program = Dsex.ChainOfThought.new("question -> answer")

      assert [{"predict", predictor}] = Dsex.Program.named_predictors(program)
      assert match?(%Dsex.Program{}, predictor)
      assert [only] = Dsex.Program.predictors(program)
      assert Dsex.Python.same?(only, predictor)
    end
  end

  describe "deepcopy/1 and reset_copy/1" do
    test "deepcopy produces an independent program" do
      program = Dsex.Predict.new("question -> answer")
      copy = Dsex.Program.deepcopy(program)

      refute copy.py == program.py
    end

    test "reset_copy clears a predictor's demos" do
      program = Dsex.Predict.new("question -> answer")
      [{_name, predictor}] = Dsex.Program.named_predictors(program)
      Dsex.Python.setattr(predictor, "demos", [%{question: "2+2?", answer: "4"}])

      reset = Dsex.Program.reset_copy(program)
      [{_name, reset_predictor}] = Dsex.Program.named_predictors(reset)

      assert Dsex.Python.getattr(reset_predictor, "demos") == []
    end
  end

  describe "set_lm/2" do
    test "overrides the program's LM independent of the surrounding context" do
      program = Dsex.Predict.new("question -> answer")
      lm = Dsex.LM.dummy([%{answer: "Paris"}])

      assert Dsex.Program.set_lm(program, lm) == :ok

      prediction = Dsex.Program.call(program, question: "What is the capital of France?")
      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "save/2 and load/2" do
    test "round-trips a predictor's demos through a JSON file" do
      program = Dsex.Predict.new("question -> answer")
      [{_name, predictor}] = Dsex.Program.named_predictors(program)
      Dsex.Python.setattr(predictor, "demos", [%{question: "2+2?", answer: "4"}])

      path =
        Path.join(
          System.tmp_dir!(),
          "dsex_program_test_#{System.unique_integer([:positive])}.json"
        )

      on_exit(fn -> File.rm(path) end)

      assert Dsex.Program.save(program, path) == :ok

      restored = Dsex.Predict.new("question -> answer")
      assert Dsex.Program.load(restored, path) == restored

      [{_name, restored_predictor}] = Dsex.Program.named_predictors(restored)
      assert Dsex.Python.getattr(restored_predictor, "demos") != []
    end
  end

  describe "dump_state/1 and load_state/2" do
    test "round-trips state through plain maps" do
      program = Dsex.Predict.new("question -> answer")
      [{_name, predictor}] = Dsex.Program.named_predictors(program)
      Dsex.Python.setattr(predictor, "demos", [%{question: "2+2?", answer: "4"}])

      state = Dsex.Program.dump_state(program)

      restored = Dsex.Predict.new("question -> answer")
      assert Dsex.Program.load_state(restored, state) == :ok

      [{_name, restored_predictor}] = Dsex.Program.named_predictors(restored)
      assert Dsex.Python.getattr(restored_predictor, "demos") != []
    end
  end

  describe "batch/3" do
    test "runs the program over multiple inputs" do
      lm = Dsex.LM.dummy([%{answer: "4"}, %{answer: "6"}])
      program = Dsex.Predict.new("question -> answer")

      results =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.batch(program, [%{question: "2+2?"}, %{question: "3+3?"}], num_threads: 1)
        end)

      assert Enum.map(results, &Dsex.Prediction.get(&1, :answer)) == ["4", "6"]
    end
  end
end
