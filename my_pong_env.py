import gym
import numpy as np
import pygame
import math

class MyPongEnv(gym.Env):
    def __init__(self):
        # Initialize your environment here
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(low=MIN_OBS, high=MAX_OBS, shape=OBS_SHAPE, dtype=np.float32) #Top left is (0,0), bottom right is (255, 255)
        self.paddle_position = INITIAL_PADDLE_POSITION #Intilization of the paddle position based on the middle of the paddle
        self.opponent_paddle_position = INITIAL_PADDLE_POSITION
        self.ball_position = { 
            'x' : INITIAL_X_BALL_POSITION,
            'y' : INITIAL_Y_BALL_POSITION
        } # Intializes the intial ball position based on the middle of the ball 
        self.ball_speed_x = BALL_SPEED_X * (np.random.randint(2) * 2 - 1) # Sets the initial ball speed with initial directional movement too
        self.ball_speed_y = BALL_SPEED_Y * (np.random.randint(2) * 2 - 1) # Sets the initial ball speed with initial directional movement too
        self.game_score = {
            'Opponent' : 0,
            'You'      : 0
            } # Sets the initial game score
        
    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH, FIELD_HEIGHT))
        pygame.display.set_caption('Custom Pong Environment')
        self.font = pygame.font.Font(None, 18)  # Font for the scoreboard

    def update_display(self):
        self.screen.fill((0, 0, 0))  # Fill the screen with black
        
        # Draw paddles
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(FIELD_WIDTH - PADDLE_WIDTH - GAP_FROM_WALL, self.paddle_position - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0 + GAP_FROM_WALL, self.opponent_paddle_position - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_position['x']), int(self.ball_position['y'])), BALL_DIAMETER / 2)
        
        pygame.display.flip()  # Update the display

    def step(self, action):
        new_observation, reward, done, info = self.take_action(action)
        return new_observation, reward, done, info
        

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

        info = {} #placeholder
        
        # Return the initial observation
        initial_observation = self.get_observation()
        initial_observation = np.array(initial_observation, dtype=np.uint8)
        return initial_observation, info

    def render(self, mode='human'):
        if mode == 'human':
            # Render the environment using a human-friendly display
            print("Rendering the Pong environment:")
            print(f"Player paddle position: {self.paddle_position}")
            print(f"Opponent paddle position: {self.opponent_paddle_position}")
            print(f"Ball position: {self.ball_position}")
            print(f"Game score: {self.game_score}")
        elif mode == 'rgb_array':
            # Return an RGB image of the environment (not implemented)
            pass
        else:
            super().render(mode=mode)

    def take_action(self, action):
        # Apply the action to the environment and calculate new state and reward
        # Update the game state based on the chosen action
        self.update_paddle_position(action, False)
        self.update_paddle_position(self.get_opponent_action(), True)
        self.update_ball_position()

        new_observation = self.get_observation()  # Calls to create the new_observation
        new_observation = new_observation.astype(np.uint8)  # Convert to uint8

        done = self.is_episode_done()  # Calls to see if the episode is done

        reward = self.calculate_reward(done)  # Calls to calculate the reward value

        info = {}  # chance for additional info

        return new_observation, reward, done, info
    
    def update_paddle_position(self, action, is_opponent):
        # if zero the paddle will move down
        if is_opponent:
            if action == 0:
                self.opponent_paddle_position = max(0 + (PADDLE_HEIGHT / 2), self.opponent_paddle_position - PADDLE_SPEED) #max makes sure that the paddle does not go out of bounds ... PADDLE_HEIGHT/2 is added because the paddle position is based of the center of the paddle
                # if 1 the paddle will move up
            elif action == 1: 
                self.opponent_paddle_position = min(FIELD_HEIGHT - (PADDLE_HEIGHT / 2), self.opponent_paddle_position + PADDLE_SPEED) #min prevents the paddle from going to low ... Same thing the subtracting the PADDLE_HEIGHT / 2

        else:
            if action == 0:
                self.paddle_position = max(0 + (PADDLE_HEIGHT / 2), self.paddle_position - PADDLE_SPEED) #max makes sure that the paddle does not go out of bounds ... PADDLE_HEIGHT/2 is added because the paddle position is based of the center of the paddle
                # if 1 the paddle will move up
            elif action == 1: 
                self.paddle_position = min(FIELD_HEIGHT - (PADDLE_HEIGHT / 2), self.paddle_position + PADDLE_SPEED) #min prevents the paddle from going to low ... Same thing the subtracting the PADDLE_HEIGHT / 2

    def update_ball_position(self):
        # Update the x position
        if abs(self.ball_speed_x) > 0:
            self.ball_position['x'] += self.ball_speed_x

            # Check for collisions with player paddle
            if 0 + GAP_FROM_WALL <= self.ball_position['x'] <= PADDLE_WIDTH + GAP_FROM_WALL:
                if self.opponent_paddle_position - PADDLE_HEIGHT / 2 <= self.ball_position['y'] <= self.opponent_paddle_position + PADDLE_HEIGHT / 2:
                    # Player paddle collision logic here
                    relative_hit_location = (self.ball_position['y'] - self.opponent_paddle_position) / (PADDLE_HEIGHT / 2)
                    bounce_angle = MAX_BOUNCE_ANGLE * relative_hit_location
                    self.ball_speed_x = BALL_SPEED * math.cos(bounce_angle)
                    self.ball_speed_y = BALL_SPEED * math.sin(bounce_angle)
                    # Move the ball out of the paddle to avoid immediate re-collision
                    self.ball_position['x'] += 1 if self.ball_speed_x > 0 else -1

            # Check for collisions with opponent paddle
            if FIELD_WIDTH - PADDLE_WIDTH - GAP_FROM_WALL <= self.ball_position['x'] <= FIELD_WIDTH - GAP_FROM_WALL:
                if self.paddle_position - PADDLE_HEIGHT / 2 <= self.ball_position['y'] <= self.paddle_position + PADDLE_HEIGHT / 2:
                    # Opponent paddle collision logic here
                    relative_hit_location = (self.ball_position['y'] - self.paddle_position) / (PADDLE_HEIGHT / 2)
                    bounce_angle = MAX_BOUNCE_ANGLE * relative_hit_location
                    self.ball_speed_x = -BALL_SPEED * math.cos(bounce_angle)  # Reverse direction for opponent
                    self.ball_speed_y = BALL_SPEED * math.sin(bounce_angle)
                    # Move the ball out of the paddle to avoid immediate re-collision
                    self.ball_position['x'] += 1 if self.ball_speed_x > 0 else -1

        # Update the y position
        if abs(self.ball_speed_y) > 0:
            self.ball_position['y'] += self.ball_speed_y

            # Check for collisions with boundaries
            if self.ball_position['y'] - (BALL_DIAMETER / 2) <= 0:
                self.ball_speed_y = abs(self.ball_speed_y)  # Reverse the vertical velocity
            elif self.ball_position['y'] + (BALL_DIAMETER / 2) >= FIELD_HEIGHT:
                self.ball_speed_y = -abs(self.ball_speed_y)  # Reverse the vertical velocity
            # Update the overall position of the ball based on step-by-step movement
            self.ball_position['x'] += self.ball_speed_x
            self.ball_position['y'] += self.ball_speed_y


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
    
    def calculate_reward(self, done):
        if done:
            if self.player_win:
                return 1
            elif self.opponent_win:
                return -1
        else:
            return 0
    
    def round_reset(self):
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

        info = {} #placeholder
        
        # Return the initial observation
        initial_observation = self.get_observation()
        initial_observation = np.array(initial_observation, dtype=np.uint8)

    
    def is_episode_done(self):
        # Check if the ball has gone out of bounds
        if self.ball_position['x'] < 0:
            self.game_score['You'] += 1
            self.opponent_win = False
            self.player_win = True
            self.round_reset()
            return True
        
        elif self.ball_position['x'] > FIELD_WIDTH:
            self.game_score['Opponent'] += 1
            self.opponent_win = True
            self.player_win = False
            self.round_reset()
            return True
        
        # Add more conditions if necessary (e.g., based on game score)
        
        return False
    
    def get_opponent_action(self):
        ball_y = self.ball_position['y']  # Ball's vertical position
        paddle_y = self.opponent_paddle_position  # Paddle's vertical position

        # Choose action based on ball and paddle positions
        if ball_y < paddle_y:
            action = 0  # Move paddle up
        else:
            action = 1  # Move paddle down
        
        return action
    




# #Constants WARNING WARNING WARNING : THE CONSTANT VALUES NEED TO BE FIGURED OUT!!!!

NUM_ACTIONS = 2  # Up and down actions
MIN_OBS = 0      # Assuming pixel values range from 0 to 255
MAX_OBS = 255
OBS_SHAPE = (84, 84, 1)  # Grayscale images with shape (height, width, num_channels)
#these need to be checked
FIELD_HEIGHT = 150 # The total height of the playing field / game window
FIELD_WIDTH = 200 # The total width of the playing field / game window
GAP_FROM_WALL = 6.25 # The distance between the paddel and the wall
PADDLE_HEIGHT = 20 # The height of the paddle
PADDLE_WIDTH = 2.5 # The width of the paddle
BALL_DIAMETER = 3.75 # The diameter of the ball 
PADDLE_SPEED = 1.25 # The speed the paddle moves 
BALL_SPEED = 1.25 # General Ball Speed
BALL_SPEED_X = 1.25 # the speed of the ball in the x direction
BALL_SPEED_Y = 1.25 # the speed of the ball in the y direction 
WHITE = (255, 255, 255) # Creation of value for the color white
MAX_BOUNCE_ANGLE = math.radians(60) # Max angle the ball can bounce off of the paddle


#Calculations

# The following calculation will calculate the vertical posistion at which the top of the paddle should be placed in order to center it.
INITIAL_PADDLE_POSITION = FIELD_HEIGHT / 2 # Calculates the inital vertical position of the paddle
# The following calculations will caluclate the horiztonal and vertical position of the ball to place it in the center of the field along the axis
INITIAL_X_BALL_POSITION = FIELD_WIDTH / 2 # Calculates the initial horizontal position of the ball 
INITIAL_Y_BALL_POSITION = FIELD_HEIGHT / 2 # Caluclates teh initial vertical position of the ball