defmodule Dsex.ExampleTest do
  use ExUnit.Case, async: true

  test "new/2 with inputs: splits into inputs/labels" do
    example = Dsex.Example.new(%{question: "2+2?", answer: "4"}, inputs: [:question])

    assert Dsex.Example.to_map(Dsex.Example.inputs(example)) == %{question: "2+2?"}
    assert Dsex.Example.to_map(Dsex.Example.labels(example)) == %{answer: "4"}
  end

  test "with_inputs/2 marks fields as inputs on an existing example" do
    example =
      %{question: "2+2?", answer: "4"}
      |> Dsex.Example.new()
      |> Dsex.Example.with_inputs([:question])

    assert Dsex.Example.to_map(Dsex.Example.inputs(example)) == %{question: "2+2?"}
  end

  test "to_map/1 returns a map with atom keys" do
    example = Dsex.Example.new(question: "2+2?", answer: "4")

    assert Dsex.Example.to_map(example) == %{question: "2+2?", answer: "4"}
  end

  test "get/2,3 reads a field or falls back to the default" do
    example = Dsex.Example.new(answer: "4")

    assert Dsex.Example.get(example, :answer) == "4"
    assert Dsex.Example.get(example, :missing) == nil
    assert Dsex.Example.get(example, :missing, "n/a") == "n/a"
  end

  test "put/3 returns a new example and leaves the original untouched" do
    original = Dsex.Example.new(answer: "4")
    updated = Dsex.Example.put(original, :answer, "5")

    assert Dsex.Example.get(updated, :answer) == "5"
    assert Dsex.Example.get(original, :answer) == "4"
  end
end
