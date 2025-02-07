from __future__ import annotations

from CybORG import CybORG
from CybORG.Agents import BaseAgent

# Import your custom agents here.
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from blue_ppo_agent import BluePPOAgent
from fileinfo_wrapper import FileInfoEnv

from gymnasium.utils import seeding

class Submission:

    # Submission name
    NAME: str = "PUNCH Submission 3"

    # Name of your team
    TEAM: str = "PUNCH"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "File Info Wrapper w/ Single Policy PPO"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": BluePPOAgent(f"blue_agent_{agent}", np_random=seeding.np_random()) for agent in range(5)
    }
    
    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return FileInfoEnv(env, pad_spaces=True)
