import numpy as np
import sys
from numpy.random import choice, randint

'Number of states'
NumberStates = 5
'Number of agents'
NumberAgents = 2
'Number of actions per agent (assumed to be equal for all agents)'
NumberActions = 2


'GENERAL DOMAIN CLASS'
class Domain:
    def __init__(self, name, States, Agents, Actions, TransitionProbs):
        self.name = name
        'Number of states'
        self.States = States
        'Number of agents'
        self.Agents = Agents
        'Dictionary mapping agents to numbers of actions'
        self.Actions = Actions
        'Transition matrices'
        self.TransitionProbs = TransitionProbs

    def NextState(self,action, state):
        return choice(range(self.States), 1, p=self.TransitionProbs[action][state])[0]

    def Sample(self,num):
        sample = []
        for i in range(num):
            initialState = randint(0,self.States)
            action = randint(0,self.Actions*self.Agents)
            nextState = self.NextState(action,initialState)
            sample.append([initialState,action,nextState])
        return sample

'RANDOM STOCHASTIC DOMAIN'
class RSD(Domain):
    def __init__(self, name, States, Agents, Actions, slope, offset):
        Domain.__init__(self, name, States, Agents, Actions, TransitionProbs= None)
        randomTrans = np.random.rand(self.Actions * self.Agents, self.States, self.States)
        print(randomTrans.shape)
        randomTrans = np.maximum(0, randomTrans * slope + offset)
        normFac = randomTrans.sum(axis=-1, keepdims=1)
        for i in range(self.Actions * self.Agents):
            for j in range(self.States):
                if normFac[i][j][0] < sys.float_info.epsilon:
                    print('NORM '+str(i))
        randomTrans = np.true_divide(randomTrans, randomTrans.sum(axis=-1, keepdims=1))
        print(randomTrans.shape)
        self.TransitionProbs = randomTrans


'Example'
rsd = RSD('RSD',
          NumberStates,
          NumberAgents,
          NumberActions,
          2,
          -1
                      )

print(rsd.TransitionProbs)
print(rsd.NextState(0,0))
sample = rsd.Sample(100)
print(sample)



'RANDOM DETERMINISTIC DOMAIN'
class RDDomain(Domain):
    def __init__(self, name, States, Agents, Actions):
        Domain.__init__(self, name, States, Agents, Actions, TransitionProbs=None)
        randomTrans = np.random.randint(0, States, size=(self.Actions * self.Agents, self.States, 1))
        self.TransitionProbs = randomTrans
    def NextState(self,action, state):
        return self.TransitionProbs[action][state][0]


'E Example'
rdd = RDDomain('RDD',
                      NumberStates,
                      NumberAgents,
                      NumberActions
                      )

print(rdd.TransitionProbs)
print(rdd.NextState(1,4))
sample = rdd.Sample(100)
print(sample)


