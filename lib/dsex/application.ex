defmodule Dsex.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Task.Supervisor, name: Dsex.TaskSupervisor},
      Dsex.Bridge
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Dsex.Supervisor)
  end
end
