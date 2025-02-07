import os

from CybORG.Agents import BaseAgent
from gym import Space
from gymnasium.spaces.multi_discrete import MultiDiscrete
from ray.rllib.models.preprocessors import OneHotPreprocessor
from ray.rllib.policy import Policy
import numpy as np

NUM_AGENTS = 5
# Set Observation Spaces, from env.observation_space('blue_agent_0') and env.observation_space('blue_agent_4')
SHORT_OBS_SPACE = MultiDiscrete([3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
LONG_OBS_SPACE = MultiDiscrete([3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

POLICY_MAP = {f"blue_agent_{i}": ("Agent0", LONG_OBS_SPACE) if i<4 else ("Agent0", LONG_OBS_SPACE) for i in range(NUM_AGENTS)}


class BluePPOAgent(BaseAgent):
    def __init__(self, name: str = None, np_random=None):
        super().__init__(name, np_random)

        assert name in POLICY_MAP

        ray_agent_name = POLICY_MAP[name][0]
        obs_space = POLICY_MAP[name][1]

        self.preprocessor = OneHotPreprocessor(obs_space=obs_space)
        self.policy = Policy.from_checkpoint(os.path.dirname(__file__) + f"/policies/{ray_agent_name}")
        self.agent_rnn_state = None

    def get_action(self, observation: dict, action_space: Space):
        transformed_obs = self.preprocessor.transform(observation)
        
        if self.agent_rnn_state is not None:
            rnn_state = self.agent_rnn_state
        else:
            rnn_state = self.policy.get_initial_state()
        
        action, rnn_state, info = self.policy.compute_single_action(transformed_obs, state=rnn_state)
        self.agent_rnn_state = rnn_state
        return action