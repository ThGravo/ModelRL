from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
import MDP_learning.multi_agent.policies as MAPolicies

scenario_name = 'simple_spread'

# load scenario from script
scenario = scenarios.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                    reward_callback=scenario.reward,
                    observation_callback=scenario.observation,
                    info_callback=None,
                    done_callback=None,
                    shared_viewer=False)
# render call to create viewer window (necessary only for interactive policies)
env.render()
# create interactive policies for each agent
# policies = [InteractivePolicy(env, i) for i in range(env.n)]
policies = [MAPolicies.RandomPolicy(env, i) for i in range(env.n)]

# print infos
print("action_space: ")
print(env.action_space)
print("discrete_action_space: ")
print(env.discrete_action_space)
print("discrete_action_input: ")
print(env.discrete_action_input)

# execution loop
obs_n = env.reset()
while True:
    # query for action from each agent's policy
    act_n = []
    for i, policy in enumerate(policies):
        act_n.append(policy.action(obs_n[i]))
    # step environment
    obs_n, reward_n, done_n, _ = env.step(act_n)
    # render all agent views
    env.render()
    # display rewards
    for agent in env.world.agents:
        print(agent.name + " reward: %0.3f" % env._get_reward(agent))
