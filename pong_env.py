import pygame
import numpy as np
import gym
from gym import spaces
import ai_paddle  # Import the AI script

# Initialize Pygame
pygame.init()


# Constants
WIDTH, HEIGHT = 800, 400
BALL_SPEED = 1.25
PADDLE_SPEED = 1
WHITE = (255, 255, 255)
left_score = 0
right_score = 0
collision_speed_boost_timer = 0
NUM_ACTIONS = 2  # Up and down actions
MIN_OBS = 0      # Assuming pixel values range from 0 to 255
MAX_OBS = 255
OBS_SHAPE = (84, 84, 1)  # Grayscale images with shape (height, width, num_channels)

# Font and text settings
font = pygame.font.Font(None, 36)  # You can adjust the font size and style
text_color = (255, 255, 255)  # White text color

# FPS tracking variables
clock = pygame.time.Clock()

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Initialize game variables
ball_pos = np.array([WIDTH // 2, HEIGHT // 2])
ball_vel = np.array([BALL_SPEED, BALL_SPEED])
paddle_height = 100
left_paddle_pos = np.array([20, HEIGHT // 2 - paddle_height // 2])
right_paddle_pos = np.array([WIDTH - 40, HEIGHT // 2 - paddle_height // 2])
paddle_vel = 0

# Create a custom Gym environment class
class PongEnv(gym.Env):
    def __init__(self):
        super(PongEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(NUM_ACTIONS)  # Define NUM_ACTIONS
        self.observation_space = spaces.Box(low=MIN_OBS, high=MAX_OBS, shape=OBS_SHAPE, dtype=np.float32)

        # Initialize other environment-specific variables
        self.state = {}  # Define the state representation
        self.reward = 0
        self.done = False

    def reset(self):
        # Reset the game state and return the initial observation
        # ... (initialize your game variables as needed) ...
        self.state = {}  # Set the initial state
        self.done = False
        return initial_observation

    def step(self, action):
        # Apply the action, update game state, calculate reward, and check if the episode is done
        # ... (update your game state based on the action) ...
        # Calculate the new observation, reward, and check for episode termination
        observation = self.state  # Update this line with your state representation
        reward = self.reward  # Update this line with your reward calculation
        done = self.done  # Update this line based on your episode termination condition

        return observation, reward, done, {}

# Register the custom Gym environment
gym.register(id='Pong-v0', entry_point='pong_env:PongEnv')

# Create the environment
env = gym.make('Pong-v0')


# Game loop
running = True
while running:
##    for event in pygame.event.get():
##        if event.type == pygame.QUIT:
##            running = False
##        elif event.type == pygame.KEYDOWN:
##            if event.key == pygame.K_UP:
##                paddle_vel = -1  # Move the paddle up
##            elif event.key == pygame.K_DOWN:
##                paddle_vel = 1  # Move the paddle down
##        elif event.type == pygame.KEYUP:
##            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
##                paddle_vel = 0  # Stop the paddle when the key is released

    # Call clock.tick() to control the frame rate (e.g., limit to 60 FPS)
    clock.tick(300)

    # Calculate the FPS after clock.tick()
    fps = int(clock.get_fps())

    # Call the AI function and pass the height of the right paddle
    ai_paddle.ai_move(ball_pos, right_paddle_pos, paddle_height, HEIGHT, PADDLE_SPEED)
    
    # Update paddle positions
    left_paddle_pos[1] += paddle_vel
    if left_paddle_pos[1] < 0:
        left_paddle_pos[1] = 0
    elif left_paddle_pos[1] > HEIGHT - paddle_height:
        left_paddle_pos[1] = HEIGHT - paddle_height

    # Ball collisions
    if ball_pos[1] <= 0 or ball_pos[1] >= HEIGHT:
        ball_vel[1] = -ball_vel[1]

    # Check for collisions with the paddles before updating ball position
    if (ball_pos[0] - 10 <= left_paddle_pos[0] + 10 and 
        left_paddle_pos[1] <= ball_pos[1] <= left_paddle_pos[1] + paddle_height):
        # Change the ball's horizontal velocity without a speed boost
        ball_vel[0] = -ball_vel[0]

    if (ball_pos[0] + 10 >= right_paddle_pos[0] - 10 and 
        right_paddle_pos[1] <= ball_pos[1] <= right_paddle_pos[1] + paddle_height):
        # Change the ball's horizontal velocity without a speed boost
        ball_vel[0] = -ball_vel[0]
        
    # Update ball position
    ball_pos = ball_pos.astype('float64') + ball_vel

    # Ball out of bounds - handle scoring logic here
    if ball_pos[0] <= 0:
        # Right paddle scores a point
        right_score += 1
        ball_pos = np.array([WIDTH // 2, HEIGHT // 2])
        ball_vel = np.array([BALL_SPEED, BALL_SPEED])
    elif ball_pos[0] >= WIDTH:
        # Left paddle scores a point
        left_score += 1
        ball_pos = np.array([WIDTH // 2, HEIGHT // 2])
        ball_vel = np.array([-BALL_SPEED, BALL_SPEED])

    # Decrease the collision_speed_boost_timer if it's active
    if collision_speed_boost_timer > 0:
        collision_speed_boost_timer -= 1

    # Clear the screen
    screen.fill((0, 0, 0))

    # Create a text surface with the current paddle speed
    speed_text = font.render(f"Speed: {PADDLE_SPEED}", True, text_color)
    # Create a text surface with the current paddle velocity
    speed_text2 = font.render(f"Velocity: {paddle_vel}", True, text_color)
    # Create a text surface with the current paddle velocity
    fps_text = font.render(f"FPS: {fps}", True, text_color)
    # Create text surfaces for left and right scores
    left_score_text = font.render(f"Left Score: {left_score}", True, text_color)
    right_score_text = font.render(f"Right Score: {right_score}", True, text_color)

    # Blit (draw) the score text surfaces onto the screen at specific positions
    screen.blit(left_score_text, (10, 10))  # Adjust the position as needed
    screen.blit(right_score_text, (WIDTH - 200, 10))  # Adjust the position as needed
    screen.blit(fps_text, (340, 10))  # Adjust the position as needed

    # Draw paddles and ball
    pygame.draw.rect(screen, WHITE, (left_paddle_pos[0], left_paddle_pos[1], 20, paddle_height))
    pygame.draw.rect(screen, WHITE, (right_paddle_pos[0], right_paddle_pos[1], 20, paddle_height))
    pygame.draw.ellipse(screen, WHITE, (ball_pos[0] - 10, ball_pos[1] - 10, 20, 20))


    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
