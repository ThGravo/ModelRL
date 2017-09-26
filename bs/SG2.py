import numpy as np
from numpy.random import choice, dirichlet
from itertools import product


'STATE SPACE'

'Number of states'
NumberStates = 10

'State class'
class State:
    def __init__(self,name):
        self.name = name

'Generate list of states'
States = []
for i in range(NumberStates):
    state = State('s'+str(i))
    States.append(state)

print('States', States, '', sep='\n')

'AGENT AND ACTION SPACES'

'Number of Agents'
NumberAgents = 3

'Number of actions per agent (assumed to be equal for all agents)'
NumberActions = 5

'Agent class'
class Agent:
    def __init__(self,name,actions):
        self.name = name
        self.actions = actions

'Generate list of agents'

Agents = []
for i in range(NumberAgents):
    agent = Agent('ag'+str(i),
                  ['ag'+str(i)+'ac'+str(j) for j in range(NumberActions)])

print('Agents', Agents, '', sep='\n')




'Generating joint action space'
JointActions = JointActionSpace(Agents)
print('Joint Action space', JointActions, '', sep='\n')

'Generating state,Joint action pairs'

StateActionPairs = product(States,JointActions)

'A dictionary mapping state,jointAction pairs to lists of transition probabilities for all states' \
'(Transition probability lists initialized randomly for all state,jointAction pairs)'

TransitionProbs = {StateActionPair: dirichlet(np.ones(NumberStates),size=1) for StateActionPair in StateActionPairs}


print('Some transition Probabilities \n')
for i in range(3):
    print((States[i],JointActions[i]),':',TransitionProbs[(States[i],JointActions[i])])



#NextState = choice(States, 1, p=TransitionProbs[StateActionPairs[0]])



'MULTI-AGENT DOMAIN'

class MultiAgentDomain:
    'Common base class for all games'

    def __init__(self, name, States, Agents, JointActions, TransitionProbs):
        self.name = name
        'Name of the domain'
        self.States = States
        'A list of states'
        self.Agents = Agents
        'A dictionary mapping state,jointAction pairs to lists of transition probabilities for all states'
        self.JointActions = JointActions
        self.TransitionProbs = TransitionProbs


    def Sample(self,state,action):
        NextState = choice(States, 1, p=TransitionProbs[(state,action)])
        return NextState

dom = MultiAgentDomain('dom',States,Agents,JointActions,TransitionProbs)

# sample = dom.Sample(States[0],JointActions[0])








