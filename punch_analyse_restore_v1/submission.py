from __future__ import annotations
from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Import your custom agents here.
from punchwrappers import CybORGWrapper
from punchagents import AnalyseRestoreAgent


class Submission:

    # Submission name
    NAME: str = "PUNCH submission 1"

    # Name of your team
    TEAM: str = "PUNCH"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "Analyse-Restore Heuristic"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": AnalyseRestoreAgent(f"blue_agent_{agent}") for agent in range(5)
    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return CybORGWrapper(env)
