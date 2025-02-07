from __future__ import annotations

from CybORG import CybORG
from CybORG.Agents import BaseAgent

# Import your custom agents here.
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from analyse_restore_agent import AnalyseRestoreAgent
from fileinfo_wrapper import FileInfoEnv

class Submission:

    # Submission name
    NAME: str = "PUNCH Submission 2"

    # Name of your team
    TEAM: str = "PUNCH"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "Round Robin Analyse and Restore w/ FileWrapper"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": AnalyseRestoreAgent(f"blue_agent_{agent}") for agent in range(5)
    }
    
    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return FileInfoEnv(env, pad_spaces=False)
