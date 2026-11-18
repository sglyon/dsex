defmodule Dsex.SignatureTest do
  use ExUnit.Case, async: true

  defmodule QA do
    use Dsex.Signature
    @moduledoc "Answer the question using the given context."

    input(:question, :string, desc: "the question to answer")
    input(:context, {:list, :string}, desc: "supporting passages")
    output(:confidence, {:literal, ["low", "medium", "high"]}, desc: "confidence level")
    output(:answer, :integer, desc: "the final answer")
  end

  @string_spec "question, context: list[str] -> " <>
                 "confidence: typing.Literal['low', 'medium', 'high'], answer: int"

  test "DSL and string-form signatures describe the same fields" do
    dsl_sig = QA.signature()
    string_sig = Dsex.Signature.new(@string_spec)

    # A plain string signature cannot carry per-field descriptions (DSPy
    # fills a placeholder like "${question}"), so compare everything else.
    without_desc = fn fields -> Enum.map(fields, &Map.delete(&1, :desc)) end

    assert without_desc.(Dsex.Signature.input_fields(dsl_sig)) ==
             without_desc.(Dsex.Signature.input_fields(string_sig))

    assert without_desc.(Dsex.Signature.output_fields(dsl_sig)) ==
             without_desc.(Dsex.Signature.output_fields(string_sig))
  end

  test "field types render as the correct Python type strings" do
    outputs_by_name = Map.new(Dsex.Signature.output_fields(QA), &{&1.name, &1})
    inputs_by_name = Map.new(Dsex.Signature.input_fields(QA), &{&1.name, &1})

    assert outputs_by_name.answer.type == "int"
    assert outputs_by_name.confidence.type == "Literal['low', 'medium', 'high']"
    assert inputs_by_name.context.type == "list[str]"
  end

  test "input_fields/output_fields preserve declaration order and descriptions" do
    assert Dsex.Signature.input_fields(QA) == [
             %{name: :question, type: "str", desc: "the question to answer", prefix: "Question:"},
             %{name: :context, type: "list[str]", desc: "supporting passages", prefix: "Context:"}
           ]
  end

  test "DSL @moduledoc becomes the signature's instructions" do
    assert Dsex.Signature.instructions(QA) == "Answer the question using the given context."
  end

  test "with_instructions/2 returns a new signature without mutating the original" do
    original = Dsex.Signature.new("question -> answer")
    updated = Dsex.Signature.with_instructions(original, "Be terse.")

    assert Dsex.Signature.instructions(updated) == "Be terse."
    refute Dsex.Signature.instructions(original) == "Be terse."
  end

  test "from/1 normalizes strings, modules, and structs to the same shape" do
    from_string = Dsex.Signature.from("question -> answer")
    from_module = Dsex.Signature.from(QA)
    from_struct = Dsex.Signature.from(from_module)

    assert Dsex.Signature.to_string(from_string) == "question -> answer"
    assert from_struct == from_module
  end

  test "signature/0 is cached across calls" do
    assert QA.signature() == QA.signature()
  end

  test "append/4 and prepend/4 add fields to the requested section" do
    base = Dsex.Signature.new("question -> answer")

    appended = Dsex.Signature.append(base, :confidence, :output, type: :float, desc: "score")
    assert Enum.map(Dsex.Signature.output_fields(appended), & &1.name) == [:answer, :confidence]

    prepended = Dsex.Signature.prepend(base, :context, :input, desc: "extra context")
    assert Enum.map(Dsex.Signature.input_fields(prepended), & &1.name) == [:context, :question]
  end

  defmodule SimpleQA do
    use Dsex.Signature
    @moduledoc false

    input(:question, :string)
    output(:answer, :string)
  end

  test "Predict with a DSL signature and DummyLM yields the expected answer" do
    lm = Dsex.LM.dummy([%{answer: "4"}])

    answer =
      Dsex.context([lm: lm], fn ->
        predictor = Dsex.Python.call("dspy.Predict", [SimpleQA.signature()])
        prediction = Dsex.Python.call(predictor, [], question: "2+2?")
        Dsex.Prediction.get(prediction, :answer)
      end)

    assert answer == "4"
  end
end
