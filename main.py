import pygame
import numpy as np
import gym
from gym import spaces
from pong_ai import PongAI  # Import the AI module

# ... (other code as before) ...

# Create the custom Gym environment
gym.register(id='MyPong-v0', entry_point='pong_env.pong_env:MyPongEnv') # Registers and locates my class in a different file
env = gym.make('MyPong-v0') # Instantiates a gym object of MyPongEnv
print(env)
# Create the AI agent with access to the action space
ai_agent = PongAI(env.action_space)

# Game loop
running = True
while running:
    # ... (other code as before) ...

    # Call the AI to choose an action based on the current observation
    action = ai_agent.choose_action(observation)

    # Apply the chosen action to the environment
    observation, reward, done, _ = env.step(action)

    # ... (other code as before) ...
