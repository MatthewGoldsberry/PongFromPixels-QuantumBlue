import numpy as np
import gym
import pygame
import pickle

# Q-Learning Constants and Configuration

# Constants for Q-table size
NUM_STATES = 8  # The different states the ball can be in, in relation to the paddle
NUM_ACTIONS = 3  # Up, Down, Still

# Learning Parameters
ALPHA = 0.3  # Learning Rate
GAMMA = 0.99  # Discount factor for future rewards
EPSILON = 0.2  # Exploration rate
BATCH_SIZE = 25
SUCCESS_NUM = 0.92  # Running mean end value to consider the current AI a working AI

# Q-table File Handling
RESUME = True  # Continue from the last checkpoint (True) or start a new Q-table (False)
FILENAME = 'qlearn_v1.p'  # Filename for the Q-table file

# Load or Create Q-table
if RESUME:
    with open(FILENAME, 'rb') as q_file:
        q_table = pickle.load(q_file) # Load Q-table from file
else:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))  # Create a new Q-table

# Instantiate Gym Environment
gym.register(id='MyPong-v0', entry_point='my_pong_package.my_pong_env2:MyPongEnv')  # Register custom Pong environment
env = gym.make('MyPong-v0')  # Create Gym environment object
observation = env.reset()  # Reset environment for a new episode


"""
    get_state():
    
    Returns:
        int: A value representing the game state.
             - 0: Ball is well above the paddle.
             - 1: Ball is somewhat above the paddle.
             - 2: Ball is in the upper half of the paddle.
             - 3: Ball is at the same height as the middle of the paddle.
             - 4: Ball is in the lower half of the paddle.
             - 5: Ball is somewhat below the paddle.
             - 6: Ball is moderately below the paddle.
             - 7: Ball is significantly below the paddle.
"""
def get_state():
   
    # Calculate the vertical distance between the ball and the paddle
    difference = env.get_ball_position('y') - env.get_your_paddle_position()

    # Determine the game state based on the calculated difference
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
    

"""
    choose_action():

    Args:
        state (int): The current state.

    Returns:
        int: The selected action based on the epsilon-greedy policy.

    Description:
        This function selects an action based on the epsilon-greedy policy, which balances exploration
        (random action with probability epsilon) and exploitation (selecting the action with the highest
        estimated Q-value with probability 1 - epsilon).

"""
def choose_action(state):

    if np.random.rand() < EPSILON:
        # Explore: Randomly choose an action
        return np.random.choice(NUM_ACTIONS)
    else:
        # Exploit: Choose the action with the highest Q-value
        return np.argmax(q_table[state])

   
"""
    learn():

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
def learn(state, action, reward, next_state, next_action):

    predicted_Qvalue = q_table[state, action]
    target_Qvalue = reward + GAMMA * q_table[next_state, next_action]

    # Update the Q-value in the Q-table
    q_table[state, action] += ALPHA * (target_Qvalue - predicted_Qvalue)


# Setup for the Pygame and Game Loop

clock = pygame.time.Clock()  # Initialize the clock for Pygame
env.init_pygame()  # Initialize the Pygame window through a function call in the env object
running = True  # Variable controlling the game loop (True = runs, False = stops)

# Variables for Learning Condition
running_reward = 0  # Running total of rewards during training
episode_num = 0  # Counter for the number of episodes

# Counter Variables for Playing Condition
opponent_score = 0  # Running total of opponent's score
player_score = 0  # Running total of player's score (AI)
opponent_wins = 0  # Running total of opponent's wins
player_wins = 0  # Running total of player's wins (AI)

# Toggle switch between AI learning and game time mode
LEARNING = True


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
    clock.tick(60)  # Limit frame rate to 60 FPS

    # Check if the Pygame window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state()  # Get the current state of the game
    action = choose_action(state)  # Choose an action based on the current state

    #env.update_paddle_position(action)  # Move the paddle based on the chosen action

    # Step the game forward and get observation, reward, episode state, and extra info
    observation, reward, done, info = env.step(action)

    # Check if an episode is finished
    if done:
        episode_num += 1  # Increment the episode counter

        # If the AI is in learning mode
        if LEARNING:
            next_state = get_state()  # Get the next state of the game
            next_action = choose_action(next_state)  # Choose the next action based on the next state

            learn(state, action, reward, next_state, next_action)  # Update the q-table based on the Q-learning algorithm

            running_reward += reward  # Monitor the training process

            # If the episode number is a multiple of the batch size
            if episode_num % BATCH_SIZE == 0:
                # Calculate the average score during the last batch of episodes
                batch_average = running_reward / BATCH_SIZE
                print('RESETTING ENVIRONMENT: Episodes %d-%d average reward was %f. Wins %f/%f.' % (
                    episode_num - (BATCH_SIZE - 1), episode_num, batch_average,
                    ((BATCH_SIZE / 2) + batch_average * (BATCH_SIZE / 2)), BATCH_SIZE))

                # If the average reward of the batch is greater or equal to the predetermined success average number
                if batch_average >= SUCCESS_NUM:
                    running = False  # Stop the game loop

                running_reward = 0  # Reset the running reward for the next batch

            if episode_num % 10 == 0:
                pickle.dump(q_table, open(FILENAME, 'wb'))  # Update the q-table in the open file every 10 episodes

        # If the AI is in game mode
        else:
            opp = 1 if reward == -1 else 0  # Determine the value to add to the opponent score based on the reward
            play = 1 if reward == 1 else 0  # Determine the value to add to the player score based on the reward
            opponent_score += opp  # Add the determined value to the opponent score
            player_score += play  # Add the determined value to the player score

            # If the opponent has scored 21 points
            if opponent_score == 21:
                opponent_wins += 1  # Increment the win counter for the opponent
                print("OPPONENT WIN: GAME SCORE: %f-%f... AI RECORD: %f-%f... RESETTING THE GAME" % (
                    opponent_score, player_score, player_wins, opponent_wins))  # Print the game score and record
                # Reset the score counter variables
                opponent_score = 0
                player_score = 0

            # If the player has scored 21 points
            elif player_score == 21:
                player_wins += 1  # Increment the win counter for the player
                print("PLAYER WIN!!! GAME SCORE: %f-%f... AI RECORD: %f-%f... RESETTING THE GAME" % (
                    player_score, opponent_score, player_wins, opponent_wins))  # Print the game score and record
                # Reset the score counter variables
                opponent_score = 0
                player_score = 0
    
    # Update the screen display
    env.update_display()  

# Once the game loop ends, print out the q-table and close Pygame
print(q_table)
pygame.quit()
