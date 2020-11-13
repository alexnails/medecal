# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
import featureExtractor

import random,util,math

""" Functions you should fill in:
        - getBestAction
        - getAction
        - update
        - incrementQValue

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Utilize self.getLegalActions(state):
        returns [action, ... ] list of all legal actions """
class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.initializeQValues()

    """ Initialize Q values to 0 """
    def initializeQValues(self):
        self.Q_values = util.Counter() # initializes a dictionary with default entries 0

    """ Return a dictionary {action: Q(state,action)} for all legal actions """
    def getQValue(self, state, action):
        return self.Q_values[(state, action)]

    """ Find legal action with highest Q(state, action) value.
        Return None if there are no legal actions. """
    def getBestAction(self, state):
        #CODE HERE
        #   HINT: Use self.getLegalActions(state)
        #   HINT: np.argmax might simplify your math
        return None

    """ With probability self.epsilon, pick random legal action.
        Otherwise, return best action.
        Return None if there are no legal actions. """
    def getAction(self, state):
        #CODE HERE
        #   HINT: Use self.getLegalActions(state)
        #   HINT: util.flipCoin(p) returns True with probability p
        #   HINT: To pick randomly from a list, use random.choice(list) """
        return None

    """ Compute new Q values given observed reward according to the Bellman update equation """
    def update(self, state, action, nextState, reward):
        """ Estimate the future rewards using Q values from the next state,
            assuming we choose the best legal action.
            Use 0 if the next state has no legal actions. """
        #CODE HERE
        #   HINT: What should you input to self.getLegalActions?
        #   HINT: Compute QValues for all legal actions, then pick highest one """
        futureRewards = 0

        """ The target Q value is the current reward plus discounted future reward. """
        #CODE HERE
        #   HINT: Use self.discount
        target_Q = 0

        """ Update our Q values according to this target. """
        self.incrementQValue(state, action, target_Q)

    """ Increment our Q function towards target value by a factor of self.alpha """
    def incrementQValue(self, state, action, target_Q):
        #CODE HERE
        #   HINT: We have a function for evaluating the current Q values
        #   HINT: Use Bellman update (with learning rate self.alpha)
        current_Q = 0
        self.Q_values[(state, action)] = 0

    """ getAction, but without random exploration """
    def getPolicy(self, state):
        return self.getBestAction(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

""" Replace Q(s,a) with Q(f), where f is a list of observable features """
class FeaturizedQAgent(PacmanQAgent):
    def getQValue(self, state, action):
        features = featureExtractor.getFeatures(state, action)
        return self.Q_values[features]

    def incrementQValue(self, state, action, target_Q):
        current_Q = self.getQValue(state, action)
        features = featureExtractor.getFeatures(state, action)
        self.Q_values[features] = (1 - self.alpha) * current_Q + self.alpha * target_Q


import numpy as np
import torch

""" Replace Q table with function Q(s,a) using neural network """
class DeepQAgent(PacmanQAgent):
    def initializeQValues(self):
        self.weights = torch.tensor(np.random.rand(featureExtractor.num_features), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weights], lr=self.alpha)

    def getQValue(self, state, action):
        features = featureExtractor.getVectorizedFeatures(state, action)
        featureTensor = torch.tensor(features, dtype=torch.double)
        current_Q = torch.dot(featureTensor, self.weights)
        return current_Q

    def incrementQValue(self, state, action, target_Q):
        current_Q = self.getQValue(state, action)
        loss = torch.nn.MSELoss(reduction='sum')(current_Q, torch.tensor(target_Q, dtype=torch.double))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()