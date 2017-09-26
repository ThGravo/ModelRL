import numpy as np
from numpy.random import choice, dirichlet
from itertools import product


'STATE SPACE'

'Number of states'
NumberStates = 50
'State objects'
class State:
    def __init__(self,name):
        self.name = name
States = ['s'+ str(i) for i in range(NumberStates)]
print('States', States, '', sep='\n')

'AGENT AND ACTION SPACES'

'Number of Agents'
NumberAgents = 4
'Number of actions available to each agent (assumed to be equal for all agents)'
NumberActions = 6
'Group of agents represented as a dictionary mapping agent names to lists of actions'
Agents = {'ag'+str(i): ['ag'+str(i)+'ac'+str(j) for j in range(NumberActions)] for i in range(NumberAgents)}
print('Agents', Agents, '', sep='\n')

'Generating joint action space'
JointActionSpace = list(product(*[Agents[a] for a in list(Agents.keys())]))
print('Joint Action space', JointActionSpace, '', sep='\n')

'Generating state,Joint action pairs'

StateActionPairs = list(product(States,JointActionSpace))

'A dictionary mapping state,jointAction pairs to lists of transition probabilities for all states \
(Transition probability lists initialized randomly for all state,jointAction pairs)'

TransitionProbs = {StateActionPair: dirichlet(np.ones(NumberStates),size=1) for StateActionPair in StateActionPairs}


print('Some transition probabilities \n')
for i in range(3):
    print((States[i],JointActionSpace[i]),':',TransitionProbs[(States[i],JointActionSpace[i])])




'MULTI-AGENT DOMAIN'

class MultiAgentDomain:

    def __init__(self, name, States, Agents, JointActions, TransitionProbs):
        self.name = name
        'Name of the domain'
        self.States = States
        'A list of states'
        self.Agents = Agents
        self.JointActions = JointActions
        self.StateActionPairs = list(product(States,JointActions))
        'A dictionary mapping state,jointAction pairs to lists of transition probabilities for all states'
        self.TransitionProbs = TransitionProbs

    def Sample(self,StateActionPair):
        prob = self.TransitionProbs[StateActionPair][0]
        NextState = choice(self.States, 1, p=prob)
        return NextState

'Generating a Domain'
dom = MultiAgentDomain('dom',States,Agents,JointActionSpace,TransitionProbs)

'Sampling the domain \
Choosing a state action pair'
StateActionPair = dom.StateActionPairs[0]

'Generating NextState'
NextState = dom.Sample(StateActionPair)
print(NextState)

'Sampling the domain to get some training examples'
examples = []
for i in range(5):
    StateActionPair = dom.StateActionPairs[i]
    NextState = dom.Sample(StateActionPair)
    example = [StateActionPair,NextState]
    examples.append(example)
    print(example)
