import numpy as np
print("test 1")
import random
print("test 2")
from pong_game import PongGame  # Import the PongGame class from the previous code
print("test 3")


print("import finished")
# Constants for Q-learning
NUM_ACTIONS = 3  # Number of possible actions (move up, stay, move down)
NUM_STATES = 2  # Number of states (ball's vertical position and velocity)
GAMMA = 0.9  # Discount factor
ALPHA = 0.1  # Learning rate
EPSILON = 0.1  # Exploration rate

# Initialize the Q-table with zeros
Q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

# Function to discretize the state
def discretize_state(ball_pos, ball_vel):
    # Define discrete intervals for the ball's position and velocity
    pos_intervals = [0, 200, 400, 600, 800]  # Adjust intervals as needed
    vel_intervals = [-5, -2, 0, 2, 5]  # Adjust intervals as needed

    # Find the interval for the ball's position
    pos_interval = 0
    for i in range(len(pos_intervals) - 1):
        if pos_intervals[i] <= ball_pos < pos_intervals[i + 1]:
            pos_interval = i
            break

    # Find the interval for the ball's velocity
    vel_interval = 0
    for i in range(len(vel_intervals) - 1):
        if vel_intervals[i] <= ball_vel < vel_intervals[i + 1]:
            vel_interval = i
            break

    return pos_interval, vel_interval

# Q-learning algorithm
def q_learning():
    game = PongGame()
    for episode in range(1000):  # Adjust the number of episodes as needed
        print("running game")
        game.run()  # Run the game
        print ("game run")
        state = discretize_state(game.ball_pos[1], game.ball_vel[1])
        total_reward = 0
        
        while game.running:
            # Choose an action using epsilon-greedy policy
            if random.uniform(0, 1) < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)  # Exploration
            else:
                action = np.argmax(Q_table[state])  # Exploitation
            
            # Take the selected action and observe the next state and reward
            game.paddle_vel = action - 1  # Map action to paddle velocity (-1, 0, 1)
            game.handle_events()
            game.update()
            
            next_state = discretize_state(game.ball_pos[1], game.ball_vel[1])
            reward = game.right_score - total_reward  # Reward is the change in score
            total_reward = game.right_score
            
            # Update Q-value using Q-learning update rule
            Q_table[state][action] += ALPHA * (reward + GAMMA * np.max(Q_table[next_state]) - Q_table[state][action])
            
            state = next_state
            
        game.running = True  # Reset the game for the next episode

# Run the Q-learning algorithm
if __name__ == "__main__":
    q_learning()