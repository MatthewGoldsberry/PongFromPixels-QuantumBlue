import numpy as np

import random

class PongAI:
    def __init__(self, action_space, q_network):
        self.action_space = action_space
        self.q_network = q_network

    def choose_action(self, observation, epsilon):
        # With probability epsilon, explore a random action
        if random.random() < epsilon:
            return self.action_space.sample()
        
        # Otherwise, exploit the Q-network to choose the best action
        state = preprocess_observation(observation)  # Preprocess the observation
        q_values = self.q_network.predict(state)  # Get Q-values for all actions
        return np.argmax(q_values)  # Choose the action with the highest Q-value
