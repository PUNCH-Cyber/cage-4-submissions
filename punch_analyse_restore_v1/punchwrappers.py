from CybORG.Agents.Wrappers import EnterpriseMAE

class CybORGWrapper(EnterpriseMAE):
    def action_space(self, agent_name: str):
        return (self._action_space[agent_name],self.env.get_observation(agent_name))