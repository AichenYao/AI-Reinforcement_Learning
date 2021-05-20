# valueIterationAgents.py
# -----------------------
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


import mdp, util
import math
from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            new_values = util.Counter()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if (len(actions) == 0):
                    new_values[state] = self.values[state]
                else:
                    max_val = -math.inf
                    for a in actions: 
                        current_val = 0
                        successors_and_probs = self.mdp.getTransitionStatesAndProbs(state, a)
                        for pair in successors_and_probs:
                            successor, prob = pair[0], pair[1]
                            reward = self.mdp.getReward(state, a, successor)
                            current_val += prob * (reward+self.discount*self.values[successor])
                        if (current_val > max_val):
                            max_val = current_val
                    new_values[state] = max_val
            for key in self.values:
                self.values[key] = new_values[key]
        return


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        successors_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        result = 0
        for pair in successors_and_probs:
            successor, prob = pair[0], pair[1]
            reward = self.mdp.getReward(state, action, successor)
            result += prob*(reward+self.discount*self.values[successor])
        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_score = -math.inf
        max_action = None
        actions = self.mdp.getPossibleActions(state)
        if (len(actions) == 0):
            return None
        for a in actions:
            q_value = self.computeQValueFromValues(state, a)
            if (q_value > max_score):
                max_score = q_value
                max_action = a
        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            index = iteration % len(states)
            state = states[index]
            actions = self.mdp.getPossibleActions(state)
            max_val = -math.inf
            if (len(actions) == 0):
                max_val = self.values[state]
            else:
                for a in actions: 
                    current_val = 0
                    successors_and_probs = self.mdp.getTransitionStatesAndProbs(state, a)
                    for pair in successors_and_probs:
                        successor, prob = pair[0], pair[1]
                        reward = self.mdp.getReward(state, a, successor)
                        current_val += prob * (reward+self.discount*self.values[successor])
                    if (current_val > max_val):
                        max_val = current_val
            self.values[state] = max_val
        return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    
    def computeQValueFromValuesHelper(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        successors_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        result = 0
        for pair in successors_and_probs:
            successor, prob = pair[0], pair[1]
            reward = self.mdp.getReward(state, action, successor)
            result += prob*(reward+self.discount*self.values[successor])
        return result
    
    def get_predecessors(self):
        states = self.mdp.getStates()
        result_dic = {}
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                successors = self.mdp.getTransitionStatesAndProbs(state, action)
                for pair in successors:
                    nextState, prob = pair[0], pair[1]
                    if (prob > 0):
                        if (nextState in result_dic):
                            result_dic[nextState].add(state)
                        else:
                            result_dic[nextState] = {state}
        return result_dic
    
    def get_diff(self, state):
        cur_val = self.values[state]
        actions = self.mdp.getPossibleActions(state)
        highest_Q = -math.inf
        for action in actions:
            cur_Q = self.computeQValueFromValuesHelper(state, action)
            if (cur_Q > highest_Q):
                highest_Q = cur_Q
        return abs(cur_val - highest_Q)

    def updateValue(self, state):
        actions = self.mdp.getPossibleActions(state)
        result = -math.inf
        for a in actions: 
            current_val = 0
            successors_and_probs = self.mdp.getTransitionStatesAndProbs(state, a)
            if (len(successors_and_probs) == 0):
                result = self.values[state]
            else:
                for pair in successors_and_probs:
                    successor, prob = pair[0], pair[1]
                    reward = self.mdp.getReward(state, a, successor)
                    current_val += prob * (reward+self.discount*self.values[successor])
                if (current_val > result):
                    result = current_val
        return result

    def runValueIteration(self):
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors_dic = self.get_predecessors()
        for state in states:
            if not (self.mdp.isTerminal(state)):
                pq.push(state, -self.get_diff(state))
        for iteration in range(self.iterations):
            if pq.isEmpty():
                return
            new_state = pq.pop()
            if not (self.mdp.isTerminal(state)):
                new_value = self.updateValue(new_state)
                self.values[new_state] = new_value
            predecessors = predecessors_dic[new_state]
            for p in predecessors:
                diff = self.get_diff(p)
                if (diff > self.theta):
                    pq.push(p, -diff)

