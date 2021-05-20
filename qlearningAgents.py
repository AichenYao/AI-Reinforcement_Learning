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
from featureExtractors import *
import random,util,math
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.Qvalues = dict()
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) in self.Qvalues:
          return self.Qvalues[(state, action)]
        else:
          self.Qvalues[(state, action)] = 0.0
          return self.Qvalues[(state, action)]
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  
          Note that if there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        max_value = -math.inf
        actions = self.getLegalActions(state)
        #case 1: terminal state
        if len(actions) == 0:
          return 0.0
        #case 2: normal cases
        for action in actions:
          new_q = self.getQValue(state, action)
          if new_q > max_value:
            max_value = new_q
        return max_value
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_value = -math.inf
        max_action = None
        actions = self.getLegalActions(state)
        #case 1: terminal state
        if len(actions) == 0:
          return max_action
        #case 2: normal cases
        for action in actions:
          new_q = self.getQValue(state, action)
          if new_q > max_value:
            max_value = new_q
            max_action = action
          if new_q == max_value:
            new_action = random.choice([max_action, action])
            max_action = new_action
        return max_action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this upon observing a
          (state => action => nextState and reward) transition.
          You should do your Q-value update here.
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        cur_q_val = self.getQValue(state, action)
        new_q_val = cur_q_val + self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState) - cur_q_val)
        self.Qvalues[(state,action)] = new_q_val
        return

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        action = None
        actions = self.getLegalActions(state)
        optimal_action = self.computeActionFromQValues(state)
        if util.flipCoin(self.epsilon):
          action = random.choice(actions)
        else:
          action = optimal_action
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
    def getValue(self, state):
        return self.computeValueFromQValues(state)

class QLearningAgentCountExploration(QLearningAgent):
    def __init__(self, k=2, **args):
        self.visitCount = util.Counter() 
        self.k = k
        QLearningAgent.__init__(self, **args)
    # Feel free to add helper functions here
   
    def computeF(self, state, action):
        u = self.getQValue(state, action)
        n = self.visitCount[(state, action)]
        new_q = u + (self.k/(n+1))
        return new_q

    def computeMaxFandAction(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  
          Note that if there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        max_value = -math.inf
        actions = self.getLegalActions(state)
        max_action = None
        #case 1: terminal state
        if len(actions) == 0:
          return (0.0, None)
        #case 2: normal cases
        for action in actions:
          new_q = self.computeF(state, action)
          if new_q > max_value:
            max_value = new_q  
            max_action = action   
        return (max_value, max_action)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this upon observing a
          (state => action => nextState and reward) transition.
          You should do your Q-value update here.
          You should update the visit count in this function 
          for the current state action pair.
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.visitCount[(state, action)] += 1
        #print(state, action, self.visitCount[(state, action)])
        cur_q_val = self.getQValue(state, action)
        sample = reward + self.discount*self.computeMaxFandAction(nextState)[0]
        new_q_val = (1 - self.alpha)*cur_q_val + self.alpha * sample
        self.Qvalues[(state,action)] = new_q_val
        return

    def getAction(self, state):
        """
          Compute the action to take in the current state. 
          Break ties randomly.
        """
        # Pick Action
        optimal_action = self.computeMaxFandAction(state)[1]
        return optimal_action
        
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
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
    
    def getWeights(self):
      return self.weights

    def getQValue(self, state, action):
      features = self.featExtractor.getFeatures(state, action)
      allFeatures = features.keys()
      # print("allFeatures:")
      # print(allFeatures)
      allWeights = self.getWeights()
      result = 0
      for key in allFeatures:
        # print("key:")
        # print(key)
        feature_value = features[key]
        # print("feature value:")
        # print(feature_value)
        cur_weight = allWeights[key]
        # print(cur_weight)
        result += cur_weight*feature_value
      return result

    def calculate_diff(self, state, action, nextState, reward):
      actions = self.getLegalActions(nextState)
      max_nextQ = -math.inf
      for a in actions:
        cur_Q = self.getQValue(nextState,a)
        if (cur_Q > max_nextQ):
          max_nextQ = cur_Q
      if (len(actions) == 0):
        max_nextQ = 0
      nowQ = self.getQValue(state,action)
      return (reward + self.discount*max_nextQ) - nowQ

    def update(self, state, action, nextState, reward):

      features = self.featExtractor.getFeatures(state, action)
      diff = self.calculate_diff(state, action, nextState, reward)
      allFeatures = features.keys()
      for cur_feature in allFeatures:
        self.weights[cur_feature] += self.alpha * diff * features[cur_feature]
      return 

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
