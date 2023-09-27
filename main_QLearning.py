import numpy as np
import gym
import pygame
import pickle

# Helps to build the q-table (controls the size)
NUM_STATES = 8 # The different states the ball can be in, in relation to the paddle
NUM_ACTIONS = 3 # Up, Down, still

# Learning Constants
ALPHA = .3 # Learning Rate 
GAMMA = .99 # Discount factor for future rewards 
EPSILON = .2 # Exploration rate 
BATCH_SIZE = 50
SUCCESS_NUM = .96 # running mean end value to consider the current AI a working AI

# Creation/ Uploading of the q-table 
RESUME = True   # Do you want to continue from the last check point? False if not, True if you do
FILENAME = 'qlearn_v5.p' # Constant for the filename that will hold the q-table

# Conditional that checks to see if you want to start a new q_table or go with a previous one
if RESUME:
    q_table = pickle.load(open(FILENAME, 'rb')) # This loads in the file containing the q_table 
else:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS)) # This creates a q_table that is that is NUM_STATES x NUM_ACTIONS and fills each spot with zero

# Instantiating the gym, pong environment 
gym.register(id='MyPong-v0', entry_point='my_pong_package.my_pong_env:MyPongEnv') # Registers and locates my class in a different file
env = gym.make('MyPong-v0') # Instantiates a gym object of MyPongEnv
observation = env.reset() # Reset the environment to start a new episode

"""
    get_state() 

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

def get_state():
   
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
    

"""
    choose_action() 

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

def choose_action(state):


    if np.random.rand() < EPSILON:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(q_table[state])
    
"""
    learn()

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

def learn(state, action, reward, next_state, next_action):

    predicted_Qvalue = q_table [state,action]
    target_Qvalue = reward + GAMMA * q_table[next_state, next_action]

    q_table[state, action] += ALPHA *(target_Qvalue - predicted_Qvalue)

# Setup for the pygame and game loop
clock = pygame.time.Clock() # Initilizes the clock that is needed for the pygame
env.init_pygame() # Initilizes the pygame window through a call to the function in the env object
running = True # This variable controls whether the game loop runs or not (True = runs, False = stops)

# Variables used in the learning condition
running_reward = 0 # keeps a running total of the rewards given during training
episode_num = 0 # coutner variable that keeps track of how many episodes have happened

# Counter Variables for the playing condition
opponent_score = 0 # running total of the opponent score
player_score = 0 # running total of the player score (AI)
opponent_wins = 0 # running total of the wins of the opponent
player_wins = 0 # running total of the wins of the player (AI)

# Toggle switch between having the AI in learning or gametime mode
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
    clock.tick(60) # Limit frame rate to 60 FPS

    # This will stop the pygame if the window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state() # makes call to the get_state() function to grab the state of the game
    action = choose_action(state) # makes call to the choose_action() function and passes state to get the action to take

    env.update_paddle_position(action) # using the action obtained from the previous line this moves the paddle accordingly

    observation, reward, done, info = env.step(action) # steps the game forward as well as returning the oberavation, reward, episode state, and extra info

    # This occurs if an episode is finished 
    if done:
        episode_num += 1 # adds to the running total of episodes that have occurred 

        # This occurs if you want the AI in learning mode
        if LEARNING:
            next_state = get_state() # makes call to the get_state() function to grab the nest state of the game
            next_action = choose_action(next_state) # makes call to the choose_action() function and passes state to get the next action to take

            learn(state, action, reward, next_state, next_action) # makes a call to the learn() function that will update the q-table 
           
            running_reward += reward # Monitoring of the training process

            # this occurs if the episode number is a multiple of the Batch size 
            if episode_num % BATCH_SIZE == 0:
                # Calculate the average score during the last batch of 25 episodes
                batch_average = running_reward / BATCH_SIZE
                print('RESETTING ENVIRONMENT: Episodes %d-%d average reward was %f. Wins %f/%f. Running Mean: %f' % (episode_num - (BATCH_SIZE-1), episode_num, batch_average, ((BATCH_SIZE/2) + batch_average*(BATCH_SIZE/2)), BATCH_SIZE, running_reward / episode_num))
                
                # This occurs is the average reward of the batch is greater or equal to the preditermined success average number
                if batch_average >= SUCCESS_NUM:
                    running = False # This stops the game loop
                
                running_reward = 0  # Reset the running reward for the next batch

            if episode_num % 10 == 0: pickle.dump(q_table, open(FILENAME, 'wb')) # This will update the q-table in the open file every 10 episodes
        
        # This occurs if the AI is in game mode
        else:
            opp = 1 if reward == -1 else 0 # This detemines the avlue to add to the opponent score based on the reward
            play = 1 if reward == 1 else 0 # This detemines the avlue to add to the player score based on the reward
            opponent_score += opp # This adds the determined value to the counter variable opponent_score
            player_score += play # This adds the determined value to the counter variable player_score

            # This occurs if the opponent has scored 21 points
            if (opponent_score == 21):
                opponent_wins += 1 # adds a win to the counter variable for the opponent
                print("OPPONENT WIN :(( GAME SCORE: %f-%f... AI RECORD: %f-%f... RESETTING THE GAME" % (opponent_score, player_score, player_wins, opponent_wins)) # prints the game score and record
                # Reset the score counter variables 
                opponent_score = 0 
                player_score = 0
            
            #This occurs if the player has scored 21 points
            elif (player_score == 21):
                player_wins += 1 # adds a win to the counter variable for the player
                print("PLAYER WIN!!! GAME SCORE: %f-%f... AI RECORD: %f-%f... RESETTING THE GAME" % (player_score, opponent_score, player_wins, opponent_wins)) # prints the game score and record
                # Reset the score counter variables 
                opponent_score = 0
                player_score = 0

    # This then finally updates the screen display
    env.update_display()

# Once the game loop ends the q-tbale is printed out and the pygame is closed down
print(q_table)
pygame.quit()
