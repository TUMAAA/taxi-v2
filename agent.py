import numpy as np
from collections import defaultdict
from enum import Enum
import os
import sys
import random
import time

class InfoVisualizer():
    @classmethod
    def visualize(cls,iState,big_state,chosen_action,chosen_actions):
        print(f"current state before choosing action: {iState}")
        print(f"Passenger state: {PassengerLocation(iState.passenger_index)}")
        print(f"big state before choosing action: {big_state}")
        print(f"chosen action: {chosen_action} {Actions(chosen_action)}")
        print(f"chosen_actions count: {Actions(0)}: {chosen_actions[0]}\
            {Actions(1)}: {chosen_actions[1]}\
            {Actions(2)}: {chosen_actions[2]}\
            {Actions(3)}: {chosen_actions[3]}\
            {Actions(4)}: {chosen_actions[4]}\
            {Actions(5)}: {chosen_actions[5]}")

class Map:
    def __init__(self):
        self.map = [
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        ]
        print(map)
        
    def draw(self,row,col):
        for row_index, text in enumerate(self.map):
            if row_index==row:
                print(text[:col*2+1]+"x"+text[col*2+2:])
            else:
                print(text)
        print("-----")
        sys.stdout.flush()
        
        
class PassengerLocation(Enum):
    R = 0
    G = 1
    Y = 2
    B = 3
    TAXI = 4
        
class InterpretableState:
    row = -1
    col = -1 
    passenger_index = -1 
    destination_index = -1
    
    def __init__(self,state):
        self.row, self.col, self.passenger_index, self.destination_index = self.decode_state(state)
        
    def __str__(self):
        return 'row:{}, col:{}, passenger_index:{}, destination_index:{}'.format(self.row,self.col,self.passenger_index,self.destination_index)
    
    def decode_state(self,state):
        out = []
        out.append(state % 4)
        state = state // 4
        out.append(state % 5)
        state = state // 5
        out.append(state % 5)
        state = state // 5
        out.append(state)
        assert 0 <= state < 5
        return reversed(out)

class BigState(Enum):
    SEARCHING = 1
    PICKEDUP = 2
    
class Actions(Enum):
    DOWN=0
    UP=1
    RIGHT=2
    LEFT=3
    PICKUP=4
    DROPOFF=5
    
    @classmethod
    def get_action_subset(cls,big_state):
        if big_state == BigState.SEARCHING: # IDEA: add big state passenger_found
            return [Actions.DOWN, Actions.UP, Actions.LEFT, Actions.RIGHT, Actions.PICKUP]
        elif big_state == BigState.PICKEDUP:
            return [Actions.DOWN, Actions.UP, Actions.LEFT, Actions.RIGHT, Actions.DROPOFF]
       
    @classmethod
    def get_action_subset_in_int(cls,big_state):
        action_subset=Actions.get_action_subset(big_state)    
        return [action.value for action in action_subset]
    
class SarsaMaxAgent:
    min_epsilon=0.01 # mainly exploitation
    epsilon=1.0 # full exploration
    alpha = 0.01
    gamma = 1
    
    big_state = BigState.SEARCHING
    
    def __init__(self, nA):
        self.Q = defaultdict(lambda: np.zeros(nA))

    def update_epsilon(self,episode_index):
        self.epsilon=max(self.epsilon/(episode_index),self.min_epsilon)
        
    def select_action(self, state):
        action_subset=Actions.get_action_subset_in_int(self.big_state)
        probs=self.get_probs(self.Q[state][np.asarray(action_subset)],self.epsilon) # accelerate by computing mask once
        chosen_action = np.random.choice(np.asarray(action_subset), p=probs)
        
        return np.random.choice(np.asarray(action_subset), p=probs)
        
    def update_Q(self,state, action, reward, next_state):        
        self.Q[state][action] = self.Q[state][action]+ \
            self.alpha*(reward + self.gamma*self.Q_update_term_expected_sarsa(next_state)-self.Q[state][action])
    
    def Q_update_term_expected_sarsa(self,next_state):
        action_subset=Actions.get_action_subset_in_int(self.big_state)
        return np.mean(self.Q[next_state][action_subset])
        
    def Q_update_term_sarsamax(self,next_state):
        action_subset=Actions.get_action_subset_in_int(self.big_state)
        return np.amax(self.Q[next_state][action_subset])
        
        
    @classmethod
    def get_probs_sample_solution(cls,state_action_values, epsilon):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        """ Taken from sample solution notebook"""
        probs = np.ones(state_action_values.size) * epsilon / state_action_values.size
        probs[np.argmax(state_action_values)] = 1 - epsilon + (epsilon / state_action_values.size)
        return probs
    
    @classmethod
    def get_probs(cls, state_action_values, epsilon):
        num_actions=state_action_values.size
        max_return = np.amax(state_action_values) # i/f all state action values equal only first index returned
        min_return = np.amin(state_action_values)
        if (max_return-min_return) == 0: # all state action values equal
            probs = np.ones(num_actions)/num_actions
        else:
            index_max = np.where(state_action_values==max_return) # IDEA: simplify by taking one max value
            probs = np.ones(num_actions)*epsilon/(num_actions-np.size(index_max))
            probs[index_max]= (1.0-epsilon)/(np.size(index_max))
        return probs
    
    def update_big_state(self,state):
        iState = InterpretableState(state)
        if iState.passenger_index == PassengerLocation.TAXI.value:
            self.big_state = BigState.PICKEDUP

class Agent:
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        #self.Q = defaultdict(lambda: np.zeros(self.nA)) # TODO: not needed
        self.chosen_actions = defaultdict(lambda: 0)
        self.agent = SarsaMaxAgent(nA)
        self.map = Map()
        
    # ANAS: added by Anas
    def update_epsilon(self,i_episode):
        self.agent.update_epsilon(i_episode)
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        old_big_state = self.agent.big_state
        self.agent.update_big_state(state)
        new_big_state = self.agent.big_state
        chosen_action = self.agent.select_action(state)
        self.chosen_actions[chosen_action]+=1
        if True:
            iState = InterpretableState(state)        
            self.map.draw(iState.row,iState.col)
            InfoVisualizer.visualize(iState,self.agent.big_state,chosen_action,self.chosen_actions)
            print(f"epsilon: {self.agent.epsilon}, alpha={self.agent.alpha}, gamma:{self.agent.gamma}")
            
            if old_big_state != new_big_state:
                time.sleep(5)
            else:
                time.sleep(0.025)
            _ = os.system('clear')
        
        return chosen_action
        
        
      
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #iState = InterpretableState(*self.decode_state(state))
        #print(iState)
        self.agent.update_Q(state, action, reward, next_state)
       
