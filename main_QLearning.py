import numpy as np
import gym
import pygame
import pickle

NUM_STATES = 8 
NUM_ACTIONS = 3 # Up, Down
ALPHA = .3 # Learning Rate 
GAMMA = .99 # Discount factor for future rewards 
EPSILON = .2 # Exploration rate 
BATCH_SIZE = 5
SUCCESS_NUM = .96 # running mean end value to consider the current AI a working AI

RESUME = True   # Do you want to continue from the last check point? False if not, True if you do
FILENAME = 'qlearn_v5.p'

# Conditional that checks to see if you want to start a new q_table or go with a previous one
if RESUME:
    q_table = pickle.load(open(FILENAME, 'rb')) # This loads in the file containing the q_table 
else:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS)) # This creates a q_table that is that is NUM_STATES x NUM_ACTIONS and fills each spot with zero

gym.register(id='MyPong-v0', entry_point='my_pong_package.my_pong_env:MyPongEnv') # Registers and locates my class in a different file
env = gym.make('MyPong-v0') # Instantiates a gym object of MyPongEnv
observation = env.reset() # Reset the environment to start a new episode

def get_state():
    """
    Calculate and return the game state based on the relative position of the ball and paddle.
    
    Returns:
        int: A value representing the game state.
             - 0: Ball is well above the paddle.
             - 1: Ball is somewhat above the paddle.
             - 2: Ball is below the paddle.
             - 3: Ball is at the same height as the paddle.
             - 4: Ball is somewhat below the paddle.
             - 5: Ball is below the paddle.
             - 6: Ball is significantly below the paddle.
             - 7: Catch-all state for other positions.
    """

    difference = env.get_ball_position('y') - env.get_your_paddle_position()

    if difference <= -env.get_paddle_height():
        return 0
    elif difference <= -env.get_paddle_height() // 2: 
        return 1
    elif difference < 0: 
        return 2 
    elif difference == 0: 
        return 3
    elif difference < env.get_paddle_height() // 2:
        return 4
    elif difference < env.get_paddle_height():
        return 5
    elif difference < env.get_paddle_height() * 3 / 2:
        return 6
    else:
        return 7
    

def choose_action(state):
    """
    Choose an action based on the epsilon-greedy policy.

    Args:
        state (int): The current state.

    Returns:
        int: The selected action based on the epsilon-greedy policy.

    Description:
        This function selects an action based on the epsilon-greedy policy, which balances exploration
        (random action with probability epsilon) and exploitation (selecting the action with the highest
        estimated Q-value with probability 1 - epsilon).

    """

    if np.random.rand() < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(q_table[state])
    

def learn(state, action, reward, next_state, next_action):
    """
    Update the Q-table based on the Q-learning algorithm.

    Args:
        state (int): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received for taking the action.
        next_state (int): The next state after taking the action.
        next_action (int): The action to be taken in the next state.

    Description:
        This function updates the Q-table using the Q-learning algorithm, which calculates the
        updated Q-value for the current state-action pair based on the received reward and the
        estimated Q-value for the next state-action pair. The learning rate (ALPHA) and the discount
        factor (GAMMA) control the weight of the update.
    """

    predicted_Qvalue = q_table [state,action]
    target_Qvalue = reward + GAMMA * q_table[next_state, next_action]

    q_table[state, action] += ALPHA *(target_Qvalue - predicted_Qvalue)


clock = pygame.time.Clock()
env.init_pygame()
running = True
running_reward = 0
episode_num = 0
denominator = 1
opponent_score = 0
player_score = 0
opponent_wins = 0
player_wins = 0
LEARNING = False

"""
    Run the main training loop for reinforcement learning.

    Description:
        This function implements the main training loop for reinforcement learning. It continuously
        interacts with the environment, chooses actions based on the learned Q-table (epsilon-greedy policy),
        updates the environment, and performs Q-learning updates. It also monitors the training progress,
        displays the environment, and saves the Q-table periodically.

    Note:
        - The loop runs at a fixed frame rate of 60 FPS (frames per second).
        - It responds to the pygame window being closed by setting the 'running' flag to False.
        - After each episode, it calculates the running mean of rewards and prints progress.
        - It saves the Q-table to a file every 10 episodes.
"""

while running:
    clock.tick(60)

    # This will stop the pygame if the window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state()
    action = choose_action(state)

    env.update_paddle_position(action)

    observation, reward, done, info = env.step(action)
    if done:
        episode_num += 1

        if LEARNING:
            next_state = get_state()
            next_action = choose_action(next_state)

            learn(state, action, reward, next_state, next_action)

            # Monitoring of the training process
            running_reward += reward

            if episode_num % BATCH_SIZE == 0:
                # Calculate the average score during the last batch of 25 episodes
                batch_average = running_reward / BATCH_SIZE
                print('RESETTING ENVIRONMENT: Episodes %d-%d average reward was %f. Wins %f/%f. Running Mean: %f' % (episode_num - (BATCH_SIZE-1), episode_num, batch_average, ((BATCH_SIZE/2) + batch_average*(BATCH_SIZE/2)), BATCH_SIZE, running_reward / episode_num))
                
                if batch_average >= SUCCESS_NUM:
                    running = False
                
                running_reward = 0  # Reset the running reward for the next batch

            if episode_num % 10 == 0: pickle.dump(q_table, open(FILENAME, 'wb'))
        
        else:
            opponent_score += 1
            player_score += 1
            if (opponent_score == 21):
                opponent_wins += 1
                print("OPPONENT WIN :(( GAME SCORE: %f-%f... AI RECORD: %f-%f... RESETTING THE GAME" % (opponent_score, player_score, player_wins, opponent_wins))
            elif (player_score == 21):
                player_wins += 1
                print("PLAYER WIN!!! GAME SCORE: %f-%f... AI RECORD: %f-%f... RESETTING THE GAME" % (player_score, opponent_score, player_wins, opponent_wins))

    
    env.update_display()

print(q_table)
pygame.quit()
