from CybORG.Agents import BaseAgent

class AnalyseRestoreAgent(BaseAgent):
    def __init__(self, name: str = None):
        super().__init__(name)

        self.agent_name = name
        self.step_count=0
        if self.agent_name == "blue_agent_4":
            self.valid_restore_actions = range(97,145)
            self.valid_analyse_actions = range(0,48)
            self.sleep_action = 148
        else:
            self.valid_restore_actions = range(33,49)
            self.valid_analyse_actions = range(0,16)
            self.sleep_action = 49
        self.index = 0

    def get_action(self, observation, action_space):
        i = self.step_count
        self.step_count += 1

        current_index = self.index

        full_cycle_len = 7

        # First Analyse
        if i%full_cycle_len == 0:
            # Analyse
            # Check if our current index is valid
            return self.valid_analyse_actions[self.index]
        elif i%full_cycle_len == 2:
            # Check if malicious files found
            mal_files_index = -48 if self.agent_name == "blue_agent_4" else -16
            mal_files = observation[mal_files_index:]
            if mal_files[current_index]:
                # If yes, then restore
                self.index = (self.index + 1) % len(self.valid_restore_actions)
                return self.valid_restore_actions[current_index]
            # If no, then move to next machine
            self.index = (self.index + 1) % len(self.valid_restore_actions)
            self.step_count = 1
            return self.valid_analyse_actions[self.index]

        # Else just Sleep
        return self.sleep_action
    
    def _validate_index(self, observation):
        valid_host_index = -96 if self.name == 'blue_agent_4' else -32
        valid_host_len = 48 if self.name == 'blue_agent_4' else 16

        valid_hosts = observation[valid_host_index:valid_host_index+valid_host_len]
        while valid_hosts[self.index] != 1:
            self.index = (self.index + 1) % len(self.valid_restore_actions)
        