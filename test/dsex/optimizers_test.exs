defmodule Dsex.OptimizersTest do
  use ExUnit.Case, async: true

  defp predict, do: Dsex.Python.call("dspy.Predict", ["question -> answer"])

  defp example(question, answer) do
    Dsex.Example.new(%{question: question, answer: answer}, inputs: [:question])
  end

  defp exact_match_metric do
    fn example, pred ->
      Dsex.Example.get(example, :answer) == Dsex.Prediction.get(pred, :answer)
    end
  end

  describe "Dsex.Optimizers.LabeledFewShot" do
    test "copies labeled trainset examples onto the student's predictor" do
      student = predict()
      trainset = [example("Q1", "A1"), example("Q2", "A2")]

      optimizer = Dsex.Optimizers.LabeledFewShot.new(k: 2)
      compiled = Dsex.Optimizer.compile(optimizer, student, trainset: trainset, sample: false)

      demos = Dsex.Python.getattr(compiled, "demos")
      assert length(demos) == 2
      assert Dsex.Python.getattr(Enum.at(demos, 0), "question") == "Q1"
      assert Dsex.Python.getattr(Enum.at(demos, 1), "question") == "Q2"
    end
  end

  describe "Dsex.Optimizers.BootstrapFewShot" do
    test "bootstraps a demo through an Elixir metric" do
      student = predict()
      trainset = [example("Q1", "A1"), example("Q2", "A2")]
      lm = Dsex.LM.dummy(List.duplicate(%{answer: "A1"}, 20))

      metric = fn _example, pred -> Dsex.Prediction.get(pred, :answer) != nil end

      optimizer =
        Dsex.Optimizers.BootstrapFewShot.new(
          metric: metric,
          max_bootstrapped_demos: 1,
          max_labeled_demos: 0,
          max_rounds: 1
        )

      compiled =
        Dsex.context([lm: lm], fn ->
          Dsex.Optimizer.compile(optimizer, student, trainset: trainset)
        end)

      demos = Dsex.Python.getattr(compiled, "demos")
      assert length(demos) == 1
    end
  end

  describe "Dsex.Optimizers.BootstrapFewShotWithRandomSearch" do
    test "runs end-to-end and keeps the best-scoring candidate" do
      student = predict()
      trainset = for i <- 1..4, do: example("Q#{i}", "Paris")
      valset = for i <- 1..2, do: example("V#{i}", "Paris")
      lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 400))

      optimizer =
        Dsex.Optimizers.BootstrapFewShotWithRandomSearch.new(
          metric: exact_match_metric(),
          max_bootstrapped_demos: 1,
          max_labeled_demos: 1,
          num_candidate_programs: 2,
          num_threads: 1
        )

      compiled =
        Dsex.context([lm: lm], fn ->
          Dsex.Optimizer.compile(optimizer, student, trainset: trainset, valset: valset)
        end)

      result =
        Dsex.context([lm: lm], fn -> Dsex.Program.call(compiled, %{question: "capital?"}) end)

      assert Dsex.Prediction.get(result, :answer) == "Paris"
    end
  end

  describe "Dsex.Optimizers.COPRO" do
    test "proposes and scores new instructions end-to-end" do
      student = predict()
      trainset = [example("capital of France?", "Paris")]

      task_lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 50))

      prompt_lm =
        Dsex.LM.dummy(
          List.duplicate(
            %{proposed_instruction: "Answer well.", proposed_prefix_for_output_field: "Answer:"},
            50
          )
        )

      optimizer =
        Dsex.Optimizers.COPRO.new(
          metric: exact_match_metric(),
          prompt_model: prompt_lm,
          breadth: 2,
          depth: 1
        )

      compiled =
        Dsex.context([lm: task_lm], fn ->
          Dsex.Optimizer.compile(optimizer, student,
            trainset: trainset,
            eval_kwargs: %{num_threads: 1, display_progress: false}
          )
        end)

      assert %Dsex.Program{} = compiled
    end
  end

  describe "Dsex.Optimizers.MIPROv2" do
    test "runs compile end-to-end with auto: nil and tiny settings" do
      trainset = for i <- 1..4, do: example("Q#{i}", "Paris")
      valset = for i <- 1..2, do: example("V#{i}", "Paris")
      lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 200))

      # MIPROv2's instruction proposer makes its own sequence of LM calls
      # (dataset summary, program/module descriptions, proposed instructions)
      # that must not interleave with the task LM's "answer" calls used for
      # bootstrapping/evaluation; give it a dedicated prompt_model scripted
      # with the proposer's exact signatures, in call order, with a few
      # extra proposal cycles duplicated as a safety margin.
      proposal_cycle = [
        %{program_description: "A question answering program."},
        %{module_description: "Answers a question."},
        %{proposed_instruction: "Answer the question with just the city name."}
      ]

      prompt_lm =
        Dsex.LM.dummy(
          ([
             %{observations: "The dataset has questions and city answers."},
             %{summary: "QA pairs."}
           ] ++
             List.duplicate(proposal_cycle, 6))
          |> List.flatten()
        )

      optimizer =
        Dsex.context([lm: lm], fn ->
          Dsex.Optimizers.MIPROv2.new(
            metric: exact_match_metric(),
            auto: nil,
            prompt_model: prompt_lm,
            num_candidates: 2,
            max_bootstrapped_demos: 1,
            max_labeled_demos: 1,
            num_threads: 1
          )
        end)

      compiled =
        Dsex.context([lm: lm], fn ->
          Dsex.Optimizer.compile(optimizer, predict(),
            trainset: trainset,
            valset: valset,
            num_trials: 2,
            minibatch: false
          )
        end)

      result =
        Dsex.context([lm: lm], fn -> Dsex.Program.call(compiled, %{question: "capital?"}) end)

      assert Dsex.Prediction.get(result, :answer) == "Paris"
    end
  end

  describe "Dsex.Optimizers.KNNFewShot" do
    test "attaches the nearest trainset example as a demo at call time" do
      trainset = [example("capital of France?", "Paris"), example("capital of Spain?", "Madrid")]
      vectorizer = fn texts -> Enum.map(texts, fn t -> [:erlang.byte_size(t) * 1.0] end) end

      optimizer = Dsex.Optimizers.KNNFewShot.new(k: 1, trainset: trainset, vectorizer: vectorizer)
      compiled = Dsex.Optimizer.compile(optimizer, predict())

      lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 2))

      result =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(compiled, %{question: "capital of France?"})
        end)

      assert Dsex.Prediction.get(result, :answer) == "Paris"
    end
  end

  describe "Dsex.Optimizers.Ensemble" do
    test "compile/2 takes a list of programs and reduces their outputs" do
      lm = Dsex.LM.dummy([%{answer: "Paris"}, %{answer: "Paris"}])

      optimizer = Dsex.Optimizers.Ensemble.new()
      ensembled = Dsex.Optimizer.compile(optimizer, [predict(), predict()])

      result =
        Dsex.context([lm: lm], fn ->
          Dsex.Program.call(ensembled, %{question: "capital of France?"})
        end)

      assert Dsex.Prediction.get(result, :answer) == "Paris"
    end
  end

  describe "Dsex.Optimizers.SIMBA" do
    test "runs end-to-end and returns an optimized program" do
      trainset = for i <- 1..4, do: example("Q#{i}", "Paris")
      lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 400))

      optimizer =
        Dsex.Optimizers.SIMBA.new(
          metric: exact_match_metric(),
          bsize: 2,
          num_candidates: 2,
          max_steps: 2,
          max_demos: 2,
          num_threads: 1
        )

      compiled =
        Dsex.context([lm: lm], fn ->
          Dsex.Optimizer.compile(optimizer, predict(), trainset: trainset)
        end)

      assert %Dsex.Program{} = compiled
    end
  end

  describe "Dsex.Optimizers.InferRules" do
    test "bootstraps demos and induces a natural-language rule" do
      trainset = for i <- 1..4, do: example("Q#{i}", "Paris")
      task_lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 400))

      # teacher_settings' lm is used both as the bootstrap-stage teacher
      # (signature "question -> answer") and, separately, as the reflection
      # LM inside RulesInductionProgram (signature "examples_text ->
      # natural_language_rules" via ChainOfThought). Key on each call's
      # unique output-field XML tag so the same DummyLM answers both
      # correctly regardless of call order.
      reflect_lm =
        Dsex.LM.dummy(%{
          "<natural_language_rules>" => %{
            plan: "p",
            natural_language_rules: "Always answer Paris."
          },
          "<answer>" => %{answer: "Paris"}
        })

      optimizer =
        Dsex.Optimizers.InferRules.new(
          metric: exact_match_metric(),
          num_candidates: 2,
          num_rules: 1,
          max_bootstrapped_demos: 1,
          teacher_settings: %{lm: reflect_lm}
        )

      compiled =
        Dsex.context([lm: task_lm], fn ->
          Dsex.Optimizer.compile(optimizer, predict(), trainset: trainset)
        end)

      assert %Dsex.Program{} = compiled
    end
  end

  describe "Dsex.Optimizers.GEPA" do
    test "converts an Elixir %{score:, feedback:} metric result to a dspy.Prediction" do
      metric = fn _gold, _pred, _trace, _pred_name, _pred_trace ->
        %{score: 0.75, feedback: "looks good"}
      end

      lm = Dsex.LM.dummy([%{answer: "ignored"}])

      optimizer =
        Dsex.context([lm: lm], fn ->
          Dsex.Optimizers.GEPA.new(metric: metric, reflection_lm: lm, max_metric_calls: 1)
        end)

      wrapped_metric = Dsex.Python.getattr(optimizer, "metric_fn")
      result = Dsex.Python.call(wrapped_metric, [nil, nil, nil, nil, nil])

      assert Dsex.Prediction.get(result, :score) == 0.75
      assert Dsex.Prediction.get(result, :feedback) == "looks good"
    end

    test "runs end-to-end with a feedback metric and tiny max_metric_calls" do
      trainset = for i <- 1..3, do: example("Q#{i}", "Paris")
      valset = for i <- 1..2, do: example("V#{i}", "Paris")

      task_lm = Dsex.LM.dummy(List.duplicate(%{answer: "Paris"}, 400))

      reflection_lm =
        Dsex.LM.dummy(
          List.duplicate(%{reasoning: "think", improved_instruction: "Answer well."}, 400)
        )

      metric = fn gold, pred, _trace, _pred_name, _pred_trace ->
        score =
          if Dsex.Example.get(gold, :answer) == Dsex.Prediction.get(pred, :answer),
            do: 1.0,
            else: 0.0

        %{score: score, feedback: "score=#{score}"}
      end

      optimizer =
        Dsex.Optimizers.GEPA.new(
          metric: metric,
          reflection_lm: reflection_lm,
          max_metric_calls: 6,
          num_threads: 1
        )

      compiled =
        Dsex.context([lm: task_lm], fn ->
          Dsex.Optimizer.compile(optimizer, predict(), trainset: trainset, valset: valset)
        end)

      assert %Dsex.Program{} = compiled
    end
  end
end
