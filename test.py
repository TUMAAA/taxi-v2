import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from agent import Actions, BigState, SarsaMaxAgent

class Tests(unittest.TestCase):

    def test_get_action_subset(self):
        self.assertCountEqual(Actions.get_action_subset(BigState.SEARCHING), 
                         [Actions.DOWN,Actions.UP,Actions.RIGHT,Actions.LEFT,Actions.PICKUP])
        
        self.assertCountEqual(Actions.get_action_subset(BigState.PICKEDUP), 
                         [Actions.DOWN,Actions.UP,Actions.RIGHT,Actions.LEFT,Actions.DROPOFF])

    
    def test_get_action_subset_in_int(self):
        self.assertCountEqual(Actions.get_action_subset_in_int(BigState.SEARCHING), 
                         [0,1,2,3,4])
        
        self.assertCountEqual(Actions.get_action_subset_in_int(BigState.PICKEDUP), 
                         [0,1,2,3,5])
    
    def test_SarsaMaxAgent_select_action(self):
        agent = SarsaMaxAgent(6)
        test_state = 400
        agent.big_state=BigState.SEARCHING
        
        agent.Q[test_state] = np.asarray([3,3,1,2,5,2]) # Action with index 4 should be most frequent
        agent.Q[test_state][Actions.DOWN.value]=3
        agent.Q[test_state][Actions.UP.value]=3
        agent.Q[test_state][Actions.LEFT.value]=1
        agent.Q[test_state][Actions.RIGHT.value]=2
        agent.Q[test_state][Actions.PICKUP.value]=5
        agent.Q[test_state][Actions.DROPOFF.value]=100 # this should be ignored when setting agent.big_state=BigState.SEARCHING
        TEST_EPSILON=0.3
        agent.epsilon=TEST_EPSILON # choose action with highest state-action value with 1-0.3 prob and all other rest with 0.3 probab
        chosen_action=[]
        NUM_RETRIES = 400
        for i in range(NUM_RETRIES):
            selected_action = agent.select_action(test_state)
            chosen_action.append(selected_action)
        self.assertAlmostEqual(np.where(np.asarray(chosen_action)==Actions.PICKUP.value)[0].size/NUM_RETRIES,(1-TEST_EPSILON),1)


    def test_SarsaMaxAgent_get_probs(self):
        agent = SarsaMaxAgent(6)
        agent.big_state=BigState.SEARCHING
        assert_array_equal(
            SarsaMaxAgent.get_probs(np.asarray([3,3,5,2,1]), 0.4), 
            np.asarray([0.1,0.1,0.6,0.1,0.1])
        )
        
        assert_array_equal(
            SarsaMaxAgent.get_probs(np.asarray([3,3,3,3,3]), 0.4), 
            np.asarray([0.2,0.2,0.2,0.2,0.2])
        )
        
        assert_almost_equal(
            SarsaMaxAgent.get_probs(np.asarray([3,3,1,1,1]), 0.3), 
            np.asarray([0.35,0.35,0.1,0.1,0.1])
        )
        
        
if __name__ == '__main__':
    unittest.main()
