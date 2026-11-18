defmodule Dsex.MixProject do
  use Mix.Project

  def project do
    [
      app: :dsex,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Dsex.Application, []}
    ]
  end

  defp deps do
    [
      {:pythonx, "~> 0.4.11"}
    ]
  end
end
