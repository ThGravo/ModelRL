from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from MDP_learning.multi_agent import make_env2

'''
| Env name in code (name in paper) |  Communication? | Competitive? | Notes |
| --- | --- | --- | --- |
| `simple.py` | N | N | Single agent sees landmark position, rewarded based on how close it gets to landmark. Not a multiagent environment -- used for debugging policies. |
| `simple_adversary.py` (Physical deception) | N | Y | 1 adversary (red), N good agents (green), N landmarks (usually N=2). All agents observe position of landmarks and other agents. One landmark is the ‘target landmark’ (colored green). Good agents rewarded based on how close one of them is to the target landmark, but negatively rewarded if the adversary is close to target landmark. Adversary is rewarded based on how close it is to the target, but it doesn’t know which landmark is the target landmark. So good agents have to learn to ‘split up’ and cover all landmarks to deceive the adversary. |
| `simple_crypto.py` (Covert communication) | Y | Y | Two good agents (alice and bob), one adversary (eve). Alice must sent a private message to bob over a public channel. Alice and bob are rewarded based on how well bob reconstructs the message, but negatively rewarded if eve can reconstruct the message. Alice and bob have a private key (randomly generated at beginning of each episode), which they must learn to use to encrypt the message. |
| `simple_push.py` (Keep-away) | N |Y  | 1 agent, 1 adversary, 1 landmark. Agent is rewarded based on distance to landmark. Adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark. So the adversary learns to push agent away from the landmark. |
| `simple_reference.py` | Y | N | 2 agents, 3 landmarks of different colors. Each agent wants to get to their target landmark, which is known only by other agent. Reward is collective. So agents have to learn to communicate the goal of the other agent, and navigate to their landmark. This is the same as the simple_speaker_listener scenario where both agents are simultaneous speakers and listeners. |
| `simple_speaker_listener.py` (Cooperative communication) | Y | N | Same as simple_reference, except one agent is the ‘speaker’ (gray) that does not move (observes goal of other agent), and other agent is the listener (cannot speak, but must navigate to correct landmark).|
| `simple_spread.py` (Cooperative navigation) | N | N | N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions. |
| `simple_tag.py` (Predator-prey) | N | Y | Predator-prey environment. Good agents (green) are faster and want to avoid being hit by adversaries (red). Adversaries are slower and want to hit good agents. Obstacles (large black circles) block the way. |
| `simple_world_comm.py` | Y | Y | Environment seen in the video accompanying the paper. Same as simple_tag, except (1) there is food (small blue balls) that the good agents are rewarded for being near, (2) we now have ‘forests’ that hide agents inside from being seen from outside; (3) there is a ‘leader adversary” that can see the agents at all times, and can communicate with the other adversaries to help coordinate the chase. |
'''

table = "|Environment Id|Observation Space|Action Space|Reward Range|NumAgents|NumLandmarks|ComDim|PosDim|ColorDim|agent_action_range|\n"  # |Local|nonDet|kwargs|
table += "|---|---|---|---|---|---|---|---|---|---|\n"

envall = ['simple', 'simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference', 'simple_speaker_listener',
          'simple_spread', 'simple_tag', 'simple_world_comm']

for e in envall:
    env = make_env2.make_env(e)
    agent_action_range = []
    agent_action_size = []
    for i, a in enumerate(env.agents):
        agent_action_range.append("{}:{}".format(i, a.u_range))
        #agent_action_size.append("{}:{}".format(i, a.))
    table += '| {}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(e,  # |{}|{}|{}|{}
                                                        env.observation_space, env.action_space, env.reward_range,
                                                        env.n, len(env.world.landmarks), env.world.dim_c,
                                                        env.world.dim_p,
                                                        env.world.dim_color, agent_action_range)
print(table)
