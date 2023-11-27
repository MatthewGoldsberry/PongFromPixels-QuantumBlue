import gym
import numpy as np
import pygame
import math
import time



"""
MyPongEnv Class:

Description:
    This class implements a simplified Pong environment for reinforcement learning. The environment simulates
    a game where the agent controls one paddle, aiming to hit the ball past the opponent's paddle. The class
    provides methods for taking actions, updating the game state, and retrieving observations.

Purpose:
    - Reinforcement learning experiments in a custom Pong environment.
    - Agent interacts with the environment through actions: moving the paddle up or down.
    - Observations include ball and paddle positions for learning and decision-making.

Methods:
    - __init__(self): Initializes the CustomPongEnvironment instance.
    - init_pygame(self): Initializes the Pygame environment for visualization.
    - update_display(self): Updates the Pygame display based on the current game state.
    - step(self, action): Takes an action, returns new observation, reward, and done status.
    - reset(self): Resets game state to initial setup, returns initial observation.
    - take_action(self, action): Applies action to environment, calculates new state, reward, and done status.
    - update_paddle_position(self, action, is_opponent=False): Updates paddle position based on action.
    - update_ball_position(self): Updates ball position based on its current speed.
    - get_observation(self): Retrieves current observation (position of ball and paddles).
    - calculate_reward(self, done): Calculates reward based on game state.
    - round_reset(self): Resets game state at end of a round.
    - get_opponent_action(self): Determines opponent's action based on ball and paddle positions.
    - get_ball_position(self, plane): Retrieves position of the ball along specified plane (x or y).
    - get_your_paddle_position(self): Retrieves position of your paddle.
    - get_paddle_height(self): Retrieves height of the paddle.

"""

class MyPongEnv(gym.Env):
    

    """
    __init__():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Description:
        This function sets up the initial state of the environment, including the action space,
        observation space, paddle positions, ball position, ball speed, and game score. It initializes
        the paddle and ball positions based on the middle of the respective objects and sets the initial
        game score to zero for both the opponent and the player.
    """
    def __init__(self):
        
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(low=MIN_OBS, high=MAX_OBS, shape=OBS_SHAPE, dtype=np.float32)
        
        # Initialization of the paddle position based on the middle of the paddle
        self.paddle_position = INITIAL_PADDLE_POSITION  
        self.opponent_paddle_position = INITIAL_PADDLE_POSITION
        
        # Initializes the initial ball position based on the middle of the ball
        self.ball_position = {
            'x': INITIAL_X_BALL_POSITION,
            'y': INITIAL_Y_BALL_POSITION
        }  

        # Sets the initial ball speed with initial directional movement
        self.ball_speed_x = BALL_SPEED_X * (np.random.randint(2) * 2 - 1)  
        self.ball_speed_y = BALL_SPEED_Y * (np.random.randint(2) * 2 - 1) 
        
        # Sets the initial game score
        self.game_score = {
            'Opponent': 0,
            'You': 0
        }  
        
    
    """
    init_pygame():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Description:
        This function initializes the Pygame environment for visualization. It initializes Pygame,
        creates a display screen with dimensions specified by FIELD_WIDTH and FIELD_HEIGHT, sets the
        window caption to 'Custom Pong Environment,' and initializes a font for the scoreboard with a
        size of 18 points.
    """
    def init_pygame(self):
        # Ensure that Pygame is properly initialized
        pygame.init()

        # Create a display screen with dimensions specified by FIELD_WIDTH and FIELD_HEIGHT
        self.screen = pygame.display.set_mode((FIELD_WIDTH, FIELD_HEIGHT))

        # Set the window caption to 'Custom Pong Environment'
        pygame.display.set_caption('Custom Pong Environment')

        # Initialize a font for the scoreboard with a size of 18 points
        self.font = pygame.font.Font(None, 18)


    """
    update_display():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Description:
        This function updates the Pygame display by filling the screen with black, drawing the paddles
        and ball with their respective positions and dimensions, and updating the display to reflect the changes.
    """
    def update_display(self):
        # Fill the screen with black
        self.screen.fill((0, 0, 0))
        
        # Draw paddles
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(FIELD_WIDTH - PADDLE_WIDTH - GAP_FROM_WALL, self.paddle_position - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0 + GAP_FROM_WALL, self.opponent_paddle_position - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_position['x']), int(self.ball_position['y'])), BALL_DIAMETER / 2)
        
        # Update the display
        pygame.display.flip()  


    """
    step():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.
        action (int): The action to be taken in the current state.

    Returns:
        tuple: A tuple containing the new observation, reward, done flag, and additional information.

    Description:
        This function executes a step in the environment by taking the specified action, updating the
        environment state, and returning the new observation, reward, done flag, and any additional information.
    """
    def step(self, action):
        new_observation, reward, done, info = self.take_action(action)
        return new_observation, reward, done, info
        

    """
    reset():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        tuple: A tuple containing the initial observation and additional information.

    Description:
        This function resets the environment to its initial state by setting the paddle positions,
        ball position, ball speed, and game score back to their initial values. It then returns the
        initial observation and additional information as a tuple.
    """
    def reset(self):
        # Reset paddle positions
        self.paddle_position = INITIAL_PADDLE_POSITION
        self.opponent_paddle_position = INITIAL_PADDLE_POSITION
        
        # Reset ball position
        self.ball_position = {
            'x': INITIAL_X_BALL_POSITION,
            'y': INITIAL_Y_BALL_POSITION
        }
        
        # Reset ball speed with random direction
        self.ball_speed_x = BALL_SPEED_X * (np.random.randint(2) * 2 - 1)
        self.ball_speed_y = BALL_SPEED_Y * (np.random.randint(2) * 2 - 1)
        
        # Reset game score
        self.game_score = {
            'Opponent': 0,
            'You': 0
        }

        info = {} # Placeholder for additional information
        
        # Return the initial observation and additional information as a tuple
        initial_observation = self.get_observation()
        initial_observation = np.array(initial_observation, dtype=np.uint8)
        return initial_observation, info


    """
    take_action():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.
        action (int): The action to be taken in the current state.

    Returns:
        tuple: A tuple containing the new observation, reward, done flag, and additional information.

    Description:
        This function executes the specified action in the environment, updates the game state based on
        the chosen action, introduces a delay of 0.01 seconds for visualization purposes, and calculates
        the new observation, reward, done flag, and any additional information. The new observation is
        then converted to uint8 before being returned as part of the tuple.
    """
    def take_action(self, action):
        # Update the player's paddle and the opponent's paddle positions based on the chosen actions
        self.update_paddle_position(action, is_opponent=False)
        self.update_paddle_position(self.get_opponent_action(), is_opponent=True)

        # Introduce a delay for visualization purposes
        time.sleep(0.01)

        # Update the position of the ball
        self.update_ball_position()

        # Obtain the new observation, convert it to uint8
        new_observation = self.get_observation().astype(np.uint8)

        # Check if the episode is done
        done = self.is_episode_done()

        # Calculate the reward value
        reward = self.calculate_reward(done)

        info = {}  # Placeholder for additional information

        # Return the new observation, reward, done flag, and additional information as a tuple
        return new_observation, reward, done, info
    

    """
    update_paddle_position():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.
        action (int): The action to be taken in the current state.
        is_opponent (bool): Flag indicating whether the paddle being updated is the opponent's.

    Description:
        This function updates the position of the player's or opponent's paddle based on the chosen action.
        For the opponent's paddle, action 0 moves it down, and action 1 moves it up. For the player's paddle,
        action 0 moves it down, action 1 does nothing, and action 2 moves it up. The paddle position is constrained
        to stay within the boundaries of the playing field.
    """
    def update_paddle_position(self, action, is_opponent = False):
        # Update the opponent's paddle position
        if is_opponent:
            if action == 0:
                # Move the opponent's paddle down, ensuring it stays within bounds
                self.opponent_paddle_position = max(0 + (PADDLE_HEIGHT / 2),
                                                    self.opponent_paddle_position - PADDLE_SPEED)
            elif action == 1:
                # Move the opponent's paddle up, ensuring it stays within bounds
                self.opponent_paddle_position = min(FIELD_HEIGHT - (PADDLE_HEIGHT / 2),
                                                    self.opponent_paddle_position + PADDLE_SPEED)
        else:
            # Update the player's paddle position
            if action == 0:
                # Move the player's paddle down, ensuring it stays within bounds
                self.paddle_position = max(0 + (PADDLE_HEIGHT / 2), self.paddle_position - PADDLE_SPEED)
            elif action == 2:
                # Move the player's paddle up, ensuring it stays within bounds
                self.paddle_position = min(FIELD_HEIGHT - (PADDLE_HEIGHT / 2), self.paddle_position + PADDLE_SPEED)


    """
    update_ball_position():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Description:
        This function updates the x and y positions of the ball based on its current speed. It checks for
        collisions with the player and opponent paddles and handles the collision logic accordingly. It also
        checks for collisions with the upper and lower boundaries of the playing field, reversing the vertical
        velocity if needed. The overall position of the ball is updated based on step-by-step movement.
    """
    def update_ball_position(self):
        # Update the x position
        if abs(self.ball_speed_x) > 0:
            self.ball_position['x'] += self.ball_speed_x

            # Check for collisions with the player's paddle
            if 0 + GAP_FROM_WALL <= self.ball_position['x'] <= PADDLE_WIDTH + GAP_FROM_WALL:
                if self.opponent_paddle_position - PADDLE_HEIGHT / 2 <= self.ball_position['y'] <= self.opponent_paddle_position + PADDLE_HEIGHT / 2:
                    # Player paddle collision logic
                    relative_hit_location = (self.ball_position['y'] - self.opponent_paddle_position) / (PADDLE_HEIGHT / 2)
                    bounce_angle = MAX_BOUNCE_ANGLE * relative_hit_location
                    self.ball_speed_x = BALL_SPEED * math.cos(bounce_angle)
                    self.ball_speed_y = BALL_SPEED * math.sin(bounce_angle)
                    # Move the ball slightly away from the paddle to avoid immediate re-collision
                    self.ball_position['x'] += 1 if self.ball_speed_x > 0 else -1

            # Check for collisions with the opponent's paddle
            if FIELD_WIDTH - PADDLE_WIDTH - GAP_FROM_WALL <= self.ball_position['x'] <= FIELD_WIDTH - GAP_FROM_WALL:
                if self.paddle_position - PADDLE_HEIGHT / 2 <= self.ball_position['y'] <= self.paddle_position + PADDLE_HEIGHT / 2:
                    # Opponent paddle collision logic
                    relative_hit_location = (self.ball_position['y'] - self.paddle_position) / (PADDLE_HEIGHT / 2)
                    bounce_angle = MAX_BOUNCE_ANGLE * relative_hit_location
                    self.ball_speed_x = -BALL_SPEED * math.cos(bounce_angle)  # Reverse direction for opponent
                    self.ball_speed_y = BALL_SPEED * math.sin(bounce_angle)
                    # Move the ball slightly away from the paddle to avoid immediate re-collision
                    self.ball_position['x'] += 1 if self.ball_speed_x > 0 else -1

        # Update the y position
        if abs(self.ball_speed_y) > 0:
            self.ball_position['y'] += self.ball_speed_y

            # Check for collisions with upper and lower boundaries
            if self.ball_position['y'] - (BALL_DIAMETER / 2) <= 0:
                self.ball_speed_y = abs(self.ball_speed_y)  # Reverse the vertical velocity
            elif self.ball_position['y'] + (BALL_DIAMETER / 2) >= FIELD_HEIGHT:
                self.ball_speed_y = -abs(self.ball_speed_y)  # Reverse the vertical velocity

            # Update the overall position of the ball based on step-by-step movement
            self.ball_position['x'] += self.ball_speed_x
            self.ball_position['y'] += self.ball_speed_y


    """
    get_observation():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        np.ndarray: The observation array containing the position of the ball and paddles.

    Description:
        This function calculates the current position of the ball, player paddle, and opponent paddle.
        It creates an observation array and normalizes its values to the range [0, 255]. The resulting
        observation is then converted to uint8 format.
    """
    def get_observation(self):
        # Calculate the position of the ball and paddles
        ball_x = self.ball_position['x']
        ball_y = self.ball_position['y']
        player_paddle_y = self.paddle_position
        opponent_paddle_y = self.opponent_paddle_position

        # Create an observation array
        observation = np.array([ball_x, ball_y, player_paddle_y, opponent_paddle_y], dtype=np.float32)

        # Normalize and scale to [0, 255]
        normalized_observation = ((observation - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)) * 255.0

        # Convert to uint8
        observation_uint8 = normalized_observation.astype(np.uint8)

        return observation_uint8
    

    """
    calculate_reward():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.
        done (bool): Indicates whether the episode is done.

    Returns:
        float: The calculated reward.

    Description:
        This function calculates the reward based on the episode status. If the episode is done,
        it checks whether the player or opponent has won and assigns a reward accordingly. If the
        episode is not done, it returns 0.
    """
    def calculate_reward(self, done):
        # Check if the episode is done
        if done:
            # Return rewards based on the winner
            if self.player_win:
                return 1  # Player wins
            elif self.opponent_win:
                return -1  # Opponent wins
        else:
            return 0  # Continue playing
    

    """
    round_reset():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        None

    Description:
        This function resets the ball and paddle positions, as well as the ball speed, at the
        beginning of a new round. It updates the direction of the ball speed based on the winner
        of the previous round.
    """
    def round_reset(self):
        # Reset positions and ball speed for a new round
        self.ball_position = { 
            'x' : INITIAL_X_BALL_POSITION,
            'y' : INITIAL_Y_BALL_POSITION
        }
        self.paddle_position = INITIAL_PADDLE_POSITION 
        self.opponent_paddle_position = INITIAL_PADDLE_POSITION

        # Reset ball speed with random direction
        direction = -1 if self.opponent_win else 1
        self.ball_speed_x = BALL_SPEED_X * direction
        self.ball_speed_y = BALL_SPEED_Y * (np.random.randint(2) * 2 - 1)

    
    """
    is_episode_done():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        bool: True if the episode is done, False otherwise.

    Description:
        This function checks whether the current episode is done by examining the ball's position.
        If the ball has gone out of bounds on the left or right side, it updates the game score and
        signals the end of the episode. Additional conditions can be added based on game rules.
    """
    def is_episode_done(self):
        # Check if the ball has gone out of bounds
        if self.ball_position['x'] < 0:
            # Increment the player's score and reset for a new round
            self.game_score['You'] += 1
            self.opponent_win = False
            self.player_win = True
            self.round_reset()
            return True
        
        elif self.ball_position['x'] > FIELD_WIDTH:
            # Increment the opponent's score and reset for a new round
            self.game_score['Opponent'] += 1
            self.opponent_win = True
            self.player_win = False
            self.round_reset()
            return True
        
        return False
    

    """
    get_opponent_action():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        int: The selected action for the opponent (0 for moving paddle up, 1 for moving paddle down).

    Description:
        This function calculates the opponent's action based on the vertical positions of the ball
        and the opponent's paddle. If the ball is above the opponent's paddle, the action is to move
        the paddle up (action 0), otherwise, the action is to move the paddle down (action 1).
    """
    def get_opponent_action(self):
        ball_y = self.ball_position['y']  # Get the vertical position of the ball
        paddle_y = self.opponent_paddle_position  # Get the vertical position of the opponent's paddle

        # Choose action based on ball and paddle positions
        if ball_y < paddle_y:
            action = 0  # Move opponent's paddle up
        else:
            action = 1  # Move opponent's paddle down
        
        return action
    

    """
    get_ball_position():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.
        plane (str): The plane along which to get the ball position ('x' or 'y').

    Returns:
        float: The position of the ball along the specified plane.

    Description:
        This function returns the position of the ball along the specified plane ('x' or 'y').
    """
    def get_ball_position(self, plane):
        return self.ball_position[plane]
    

    """
    get_your_paddle_position():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        float: The vertical position of your paddle.

    Description:
        This function returns the vertical position of your paddle.
    """
    def get_your_paddle_position(self):
        return self.paddle_position
    

    """
    get_paddle_height():

    Args:
        self (MyPongEnv): An instance of the MyPongEnv class.

    Returns:
        float: The height of the paddles.

    Description:
        This function returns the height of the paddles.
    """
    def get_paddle_height(self):
        return PADDLE_HEIGHT



# Constants

NUM_ACTIONS = 2  # Number of actions (up and down)
MIN_OBS = 0  # Minimum pixel value
MAX_OBS = 255  # Maximum pixel value
OBS_SHAPE = (84, 84, 1)  # Grayscale image shape (height, width, channels)

# Field Dimensions
FIELD_HEIGHT = 150  # Total height of the playing field
FIELD_WIDTH = 210  # Total width of the playing field
GAP_FROM_WALL = 6.25  # Distance between the paddle and the wall
PADDLE_HEIGHT = 20  # Height of the paddle
PADDLE_WIDTH = 2.5  # Width of the paddle
BALL_DIAMETER = 3.75  # Diameter of the ball
PADDLE_SPEED = 1.25  # Speed at which the paddle moves
BALL_SPEED = 1.35  # General ball speed
BALL_SPEED_X = 1.35  # Speed of the ball in the x-direction
BALL_SPEED_Y = 1.35  # Speed of the ball in the y-direction
WHITE = (255, 255, 255)  # RGB value for the color white
MAX_BOUNCE_ANGLE = math.radians(60)  # Maximum angle for ball bounce off the paddle

# Calculations

# Initial Paddle Position (Centered vertically)
INITIAL_PADDLE_POSITION = FIELD_HEIGHT / 2

# Initial Ball Position (Centered horizontally and vertically)
INITIAL_X_BALL_POSITION = FIELD_WIDTH / 2
INITIAL_Y_BALL_POSITION = FIELD_HEIGHT / 2
