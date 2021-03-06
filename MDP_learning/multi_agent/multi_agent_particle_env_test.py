import MDP_learning.multi_agent.policies as MAPolicies
import MDP_learning.multi_agent.make_env2 as make_env

scenario_names = ['simple', 'simple_adversary',  # 'simple_crypto',
                  'simple_push', 'simple_reference', 'simple_speaker_listener',
                  'simple_spread', 'simple_tag', 'simple_world_comm']
for scenario_name in scenario_names:
    env = make_env.make_env(scenario_name)

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
    for _ in range(1000):
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
        if any(done_n):
            print("DONE")