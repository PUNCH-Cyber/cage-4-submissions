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
from CybORG.Simulator.Actions import Sleep


class FileInfoEnv(EnterpriseMAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_hosts = {}
        for agent in self.agents:
            hosts = [h for h in self.hosts(agent) if 'router' not in h]
            analyse_actions_hosts = [l.split(" ") for l in self._action_space[agent]['labels'] if 'Analyse' in l]
            valid_host_map = {h[-1]: True if len(h) == 2 else False for h in analyse_actions_hosts}
            self.valid_hosts[agent] = np.array([valid_host_map[h] for h in hosts], dtype=int)
        self.fix_action_padding()

    def step(self, action_dict: dict[str, Any] | None = None, messages: dict[str, Any] | None = None) -> tuple[dict[str, ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        obs, rew, term, trunc, info = super().step(action_dict, messages)

        new_obs = deepcopy(obs)
        for agent in new_obs:
            # Get Success value
            raw_obs = self.env.get_observation(agent)
            
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
            hosts = [h for h in self.hosts(agent) if 'router' not in h]
            target_len = MAX_HOSTS * 3 if agent == 'blue_agent_4' or self._pad_spaces else MAX_HOSTS
            pad_len = target_len - len(hosts)

            malicious_files = {h:False for h in hosts}
            mal_files_obs = [malicious_files[h] for h in hosts]
            mal_files_obs.extend([False] * pad_len)
            mal_files_obs = np.array(mal_files_obs, dtype=int)
            new_obs[agent] = append(new_obs[agent], mal_files_obs)

        self.fix_action_padding()
        return new_obs, info

    def get_observation(self, agent: str):
        obs = super().get_observation(agent)
        new_obs = deepcopy(obs)

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
        new_space = MultiDiscrete(list(start_space.nvec) + [2] * num_hosts)
        return new_space
    
    def fix_action_padding(self) -> None:
        """Pad all agent action spaces to match the size of the largest action space"""
        if not self._pad_spaces:
            return

        small_analyse_start = 0
        small_monitor_start = 16
        small_remove_start = 17
        small_restore_start = 33
        small_sleep_start = 49
        small_allow_start = 50
        small_block_start = 58
        small_decoy_start = 66
        small_decoy_end = 82
        
        large_analyse_start = 0
        large_monitor_start = 48
        large_remove_start = 49
        large_restore_start = 97
        large_sleep_start = 145
        large_allow_start = 146
        large_block_start = 170
        large_decoy_start = 194   
        
        def copy_actions(old_space, new_space, small_start, small_end, large_start):
            for key in ['actions', 'labels', 'mask']:
                for i in range(small_end-small_start):
                    new_space[key][large_start+i] = old_space[key][small_start+i]

        for agent_name in self.agents:
            space_size = len(self._action_space[agent_name]["actions"])
            pad_size = self._max_act_space_size - space_size

            # if pad_size == 0:
            if agent_name == 'blue_agent_4':
                continue

            # Setup new action space that is all padding
            old_space = deepcopy(self._action_space[agent_name])
            self._action_space[agent_name]["actions"] = [Sleep()] * self._max_act_space_size
            self._action_space[agent_name]["labels"] = ["[Padding] Sleep"] * self._max_act_space_size
            self._action_space[agent_name]["mask"] = [False] * self._max_act_space_size
            
            # Copy over the previous actions into the correct places
            copy_actions(old_space, self._action_space[agent_name], small_analyse_start, small_monitor_start, large_analyse_start)
            copy_actions(old_space, self._action_space[agent_name], small_monitor_start, small_remove_start, large_monitor_start)
            copy_actions(old_space, self._action_space[agent_name], small_remove_start, small_restore_start, large_remove_start)
            copy_actions(old_space, self._action_space[agent_name], small_restore_start, small_sleep_start, large_restore_start)
            copy_actions(old_space, self._action_space[agent_name], small_sleep_start, small_allow_start, large_sleep_start)
            copy_actions(old_space, self._action_space[agent_name], small_allow_start, small_block_start, large_allow_start)
            copy_actions(old_space, self._action_space[agent_name], small_block_start, small_decoy_start, large_block_start)
            copy_actions(old_space, self._action_space[agent_name], small_decoy_start, small_decoy_end, large_decoy_start)
