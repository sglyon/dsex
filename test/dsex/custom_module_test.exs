defmodule Dsex.CustomModuleTest.RAG do
  @moduledoc false
  use Dsex.Module

  predictor(:generate, Dsex.ChainOfThought, "context, question -> answer")

  def forward(prog, %{question: question}) do
    context = "Paris is the capital of France."
    Dsex.Program.call(prog.generate, context: context, question: question)
  end
end

defmodule Dsex.CustomModuleTest do
  use ExUnit.Case, async: true

  alias Dsex.CustomModuleTest.RAG

  describe "forward/2" do
    test "runs the declared sub-program and returns its prediction" do
      lm = Dsex.LM.dummy([%{plan: "France's capital is Paris.", answer: "Paris"}])
      program = RAG.new()

      prediction =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(program, question: "What is the capital of France?")
        end)

      assert Dsex.Prediction.get(prediction, :answer) == "Paris"
    end
  end

  describe "named_predictors/1" do
    test "lists the declared sub-program" do
      program = RAG.new()

      assert [{"generate.predict", predictor}] = Dsex.Program.named_predictors(program)
      assert match?(%Dsex.Program{}, predictor)
    end

    test "a deep copy's accessor exposes its own predictor instance" do
      program = RAG.new()
      copy = Dsex.Program.deepcopy(program)

      [{_name, original_predictor}] = Dsex.Program.named_predictors(program)
      [{_name, copy_predictor}] = Dsex.Program.named_predictors(copy)

      refute original_predictor.py == copy_predictor.py
    end
  end

  describe "save/2 and load/2" do
    test "round-trips the declared predictor's demos" do
      program = RAG.new()
      [{_name, predictor}] = Dsex.Program.named_predictors(program)

      Dsex.Python.setattr(predictor, "demos", [
        %{
          context: "Paris is the capital of France.",
          question: "capital of France?",
          answer: "Paris"
        }
      ])

      path =
        Path.join(
          System.tmp_dir!(),
          "dsex_custom_module_test_#{System.unique_integer([:positive])}.json"
        )

      on_exit(fn -> File.rm(path) end)

      assert Dsex.Program.save(program, path) == :ok

      restored = RAG.new()
      assert Dsex.Program.load(restored, path) == restored

      [{_name, restored_predictor}] = Dsex.Program.named_predictors(restored)
      assert Dsex.Python.getattr(restored_predictor, "demos") != []
    end
  end

  describe "optimization" do
    test "BootstrapFewShot compiles the custom module's predictor with demos" do
      lm =
        Dsex.LM.dummy([
          %{plan: "Known fact.", answer: "Paris"},
          %{plan: "Known fact.", answer: "Lyon"}
        ])

      metric = fn _example, prediction, _trace ->
        Dsex.Prediction.get(prediction, :answer) != nil
      end

      trainset = [
        Dsex.Example.new(%{question: "Capital of France?", answer: "Paris"},
          inputs: [:question]
        ),
        Dsex.Example.new(%{question: "Second city of France?", answer: "Lyon"},
          inputs: [:question]
        )
      ]

      program = RAG.new()

      compiled =
        Dsex.context([lm: lm], fn ->
          optimizer =
            Dsex.Python.call("dspy.BootstrapFewShot", [],
              metric: metric,
              max_bootstrapped_demos: 2,
              max_labeled_demos: 0
            )

          Dsex.Python.call_method(optimizer, "compile", [program], trainset: trainset)
        end)

      [{_name, predictor}] = Dsex.Program.named_predictors(compiled)
      demos = Dsex.Python.getattr(predictor, "demos")

      assert demos != []
    end
  end
end
