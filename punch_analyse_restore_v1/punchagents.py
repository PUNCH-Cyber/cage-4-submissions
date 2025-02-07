from CybORG.Agents import BaseAgent


class AnalyseRestoreAgent(BaseAgent):
    def __init__(self, name: str = None):
        super().__init__(name)

        self.agent_name = name
        self.agent_switch = 0
        self.agent_cooldown = 0
        self.subnet_index = 0 
        self.step = 0
        self.initial_hostnames = None

    def get_action(self, observation: dict, action_space):
        if self.step >498:
            self.agent_switch = 0
            self.agent_cooldown = 0
            self.subnet_index = 0 
            self.step = 0
            self.initial_hostnames = None
        self.step +=1         
        observation = action_space[1]
        action_space = action_space[0]
        action_names = [str(x) for x in action_space['actions']]
        sleep = [i for i,e in enumerate(action_space['labels']) if str(e) == 'Sleep'][0]
        hostnames = [str(x).split(' ')[1] for x in action_names if 'Analyse' in str(x)]
        if self.initial_hostnames is None:
            self.initial_hostnames = hostnames
        current_action = None
        if self.agent_switch == 0:
            current_action = [i for i,name in enumerate(action_names) if 'Analyse' in name and hostnames[self.subnet_index] in name][0]
            self.agent_switch = 1
        elif self.agent_switch == 1: # Waiting for Analyse to finish
            self.agent_switch = 2
            current_action = sleep
        elif self.agent_switch == 2:            
            if len(list(observation.keys())) > 2: # Analyse returned something
                ids = list(observation.keys())
                [ids.remove(x) for x in ['success','action','message'] if x in ids]
                ids = [x for x in ids if 'router' not in x]
                if len(ids) > 0:
                    for host in ids:

                        if host == hostnames[self.subnet_index] and observation[host].get('Files') is not None:             
                            current_action = [i for i,name in enumerate(action_names) if 'Restore' in name and host in name][0]
                            self.agent_switch = 3
                    if self.agent_switch != 3:
                        self.subnet_index = (self.subnet_index + 1) % len(hostnames)
                        current_action = [i for i,name in enumerate(action_names) if 'Analyse' in name and hostnames[self.subnet_index] in name][0]
                        self.agent_switch = 1
                else:
                    self.subnet_index = (self.subnet_index + 1) % len(hostnames)
                    current_action = [i for i,name in enumerate(action_names) if 'Analyse' in name and hostnames[self.subnet_index] in name][0]
                    self.agent_switch = 1  
            else: # Analyse returned nothing
                self.subnet_index = (self.subnet_index + 1) % len(hostnames)
                current_action = [i for i,name in enumerate(action_names) if 'Analyse' in name and hostnames[self.subnet_index] in name][0]
                self.agent_switch = 1 
        elif self.agent_switch >=3: #waiting for Restore to finish
            if self.agent_switch ==6:
                self.agent_switch = 0
                self.subnet_index = (self.subnet_index + 1) % len(hostnames)
                current_action = sleep
            else:
                self.agent_switch +=1
                current_action = sleep
        return current_action