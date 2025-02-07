# PUNCH CAGE 4 Agent Submissions

This repository contains the agents we submitted to the [TTCP CAGE Challenge 4](https://github.com/cage-challenge/cage-challenge-4) autonomous cyber defense competition.

Our top performing agent, which performed much better than our RL agents, applied an identical strategy in each subnet. This followed a simple pattern of analyzing and subsequently restoring infected machines in a round-robin fashion.

To implement this heuristic policy, an additional wrapper was needed that included the results from analyze actions in the observation vector. The new wrapper queried CybORG for each agent after stepping the environment and checked for files present in the raw CybORG observation. If files were present and any had a density of greater than 0.9, a feature for the host the files were from was set to **True**; else, in all other instances, it was set to **False**.

With this new information, we were able to implement the described heuristic as a state machine by tracking the step count and current state.

# Agent Details

We submitted **three different agents** to the competition. Details are provided below, and the submission bundles are included in this repository.

## Analyse Restore v1

For all three of our agents, we had to address the issue that results from the **Analyze** action (i.e., detection of malicious files) were not included in the default `EnterpriseMAE` wrapper.

For this agent, we modified the action space object passed to the agent to include the raw observation. This way, the agent could work with both the original observation and the extra information.

**Implementation:** [./punch_analyse_restore_v1](./punch_analyse_restore_v1)

### Results
- **Final score:** -142.72  
- **Placement:** 3rd

## Analyse Restore v2

For this agent, we created the `FileInfoEnv`, a wrapper that extended the original `EnterpriseMAE` wrapper. We did this to ensure we were passing observations through the environment as expected. This wrapper extends the original vector observation to include a field for each host in the subnet indicating whether or not malicious files were detected on the host.

The agent could then operate fully on the observation without needing to consult the action space as well. It implemented the same round-robin analyze and restore loop as v1.

**Implementation:** [./punch_analyse_restore_v2](./punch_analyse_restore_v2)

### Results
- **Final score:** -167.25  
- **Placement:** 4th

## PPO w/ File Info

For our RL agent, we extended the `FileInfoEnv` environment further and then trained a PPO policy on top of it. Our agent used a single policy for all subnets by leveraging observation and action space padding capabilities.

### Action Space Padding
One key addition to the wrapper was fixing the action padding that occurred with the default wrappers. By default, the actions are not always in the same position when using action padding. Specifically, to pad the action space in small subnets, padded `Sleep` actions are added at the end of the original action space. This results in an action space of the correct size; however, the larger action space is not aligned across smaller and larger agents. Each action type (e.g., **Analyze**, **Remove**, etc.) has, for example, 16 entries in the small agents and 48 entries in the large agents, with the action space sorted by action type.

For example, when padding spaces, the first four small agents (**Blue Agent 0-3**) all have a **Monitor** action at index 16, whereas **Blue Agent 4** has another **Analyze** action.

We fixed this by manually identifying the correct indexes where each action type should start and sorting the padded action space accordingly.

### Training
We trained our agents using the [RLLib PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo) implementation.

**Implementation:** [./punch_file_info_ppo](./punch_file_info_ppo)

### Results
- **Final score:** -1113.1  
- **Placement:** Would have placed 11th

