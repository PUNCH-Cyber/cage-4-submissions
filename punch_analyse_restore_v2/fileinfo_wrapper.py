from __future__ import annotations

import functools
from copy import deepcopy
from typing import Any
import numpy as np
from gymnasium import Space
from numpy import ndarray, append
from gymnasium.spaces import MultiDiscrete
from CybORG.Agents.Wrappers.BlueFlatWrapper import MAX_HOSTS
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE


class FileInfoEnv(EnterpriseMAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_hosts = {}
        for agent in self.agents:
            hosts = [h for h in self.hosts(agent) if 'router' not in h]
            analyse_actions_hosts = [l.split(" ") for l in self._action_space[agent]['labels'] if 'Analyse' in l]
            valid_host_map = {h[-1]: True if len(h) == 2 else False for h in analyse_actions_hosts}
            self.valid_hosts[agent] = np.array([valid_host_map[h] for h in hosts], dtype=int)

    def step(self, action_dict: dict[str, Any] | None = None, messages: dict[str, Any] | None = None) -> tuple[dict[str, ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        obs, rew, term, trunc, info = super().step(action_dict, messages)

        new_obs = deepcopy(obs)
        for agent in new_obs:
            # Get Success value
            raw_obs = self.env.get_observation(agent)

            new_obs[agent] = append(new_obs[agent], self.valid_hosts[agent])
            # Check for any malicious files found during Analyse
            hosts = [h for h in self.hosts(agent) if 'router' not in h]
            target_len = MAX_HOSTS * 3 if agent == 'blue_agent_4' or self._pad_spaces else MAX_HOSTS
            pad_len = target_len - len(hosts)

            malicious_files = {h:False for h in hosts}
            
            for k,v in raw_obs.items():
                if k not in ['success', 'action', 'message']:
                    if 'Files' in v:
                        for f in v['Files']:
                            if 'Density' in f and f['Density'] == 0.9:
                                malicious_files[k] = True
        
            mal_files_obs = [malicious_files[h] for h in hosts]
            mal_files_obs.extend([False] * pad_len)
            mal_files_obs = np.array(mal_files_obs, dtype=int)

            new_obs[agent] = append(new_obs[agent], mal_files_obs)
        return new_obs, rew, term, trunc, info
    
    def reset(
        self, agent=None, seed=None, *args, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(agent, seed, *args, **kwargs)

        new_obs = deepcopy(obs)
        for agent in new_obs:
            new_obs[agent] = append(new_obs[agent], self.valid_hosts[agent])

            hosts = [h for h in self.hosts(agent) if 'router' not in h]
            target_len = MAX_HOSTS * 3 if agent == 'blue_agent_4' or self._pad_spaces else MAX_HOSTS
            pad_len = target_len - len(hosts)

            malicious_files = {h:False for h in hosts}
            mal_files_obs = [malicious_files[h] for h in hosts]
            mal_files_obs.extend([False] * pad_len)
            mal_files_obs = np.array(mal_files_obs, dtype=int)
            new_obs[agent] = append(new_obs[agent], mal_files_obs)

        return new_obs, info

    def get_observation(self, agent: str):
        obs = super().get_observation(agent)
        new_obs = deepcopy(obs)

        new_obs = append(new_obs, self.valid_hosts[agent])

        hosts = [h for h in self.hosts(agent) if 'router' not in h]
        target_len = MAX_HOSTS * 3 if agent == 'blue_agent_4' or self._pad_spaces else MAX_HOSTS
        pad_len = target_len - len(hosts)
        malicious_files = {h:False for h in hosts}
        mal_files_obs = [malicious_files[h] for h in hosts]
        mal_files_obs.extend([False] * pad_len)
        mal_files_obs = np.array(mal_files_obs, dtype=int)
        new_obs = append(new_obs, mal_files_obs)

        return new_obs

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name: str) -> Space:
        """Add a new field to the end of the obs space to represent the success portion of the observation"""
        start_space = self._observation_space[agent_name]
        num_hosts = MAX_HOSTS * 3 if agent_name == 'blue_agent_4' or self._pad_spaces else MAX_HOSTS
        new_space = MultiDiscrete(list(start_space.nvec) + [2] * num_hosts + [2] * num_hosts)
        return new_space