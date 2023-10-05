print("pong Test")
import pygame
import numpy as np
import gym
from gym import spaces

print("pong env")

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

print("pong Constants")

# Font and text settings
font = pygame.font.Font(None, 36)  # You can adjust the font size and style
text_color = (255, 255, 255)  # White text color
print("pong font")

# FPS tracking variables
clock = pygame.time.Clock()
clock.tick(300)
fps = int(clock.get_fps())
print("pong FPS")

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")
print("pong Screen")

# Initialize game variables
ball_pos = np.array([WIDTH // 2, HEIGHT // 2])
ball_vel = np.array([BALL_SPEED, BALL_SPEED])
paddle_height = 100
left_paddle_pos = np.array([20, HEIGHT // 2 - paddle_height // 2])
right_paddle_pos = np.array([WIDTH - 40, HEIGHT // 2 - paddle_height // 2])
paddle_vel = 0
print("pong Variables")

def initialize_pygame():
    print("pong Initializing")
    # Register the custom Gym environment
    gym.register(id='Pong-v0', entry_point='pong_env:PongEnv')

    # Create the environment
    env = gym.make('Pong-v0')

    # Initialize Pygame
    pygame.init()

# Create a custom Gym environment class


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                paddle_vel = -1  # Move the paddle up
            elif event.key == pygame.K_DOWN:
                paddle_vel = 1  # Move the paddle down
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                paddle_vel = 0  # Stop the paddle when the key is released
    if ai_action == "up":
        paddle_vel = -1
    elif ai_action == "down":
        paddle_vel = 1

    return paddle_vel

def update_paddle_position():
    # Update paddle positions
    left_paddle_pos[1] += paddle_vel
    if left_paddle_pos[1] < 0:
        left_paddle_pos[1] = 0
    elif left_paddle_pos[1] > HEIGHT - paddle_height:
        left_paddle_pos[1] = HEIGHT - paddle_height

def handle_ball_collisions():
    # Ball collisions
    if ball_pos[1] <= 0 or ball_pos[1] >= HEIGHT:
        ball_vel[1] = -ball_vel[1]

def check_paddle_collisions():
    # Check for collisions with the paddles before updating ball position
    if (ball_pos[0] - 10 <= left_paddle_pos[0] + 10 and 
        left_paddle_pos[1] <= ball_pos[1] <= left_paddle_pos[1] + paddle_height):
        # Change the ball's horizontal velocity without a speed boost
        ball_vel[0] = -ball_vel[0]

    if (ball_pos[0] + 10 >= right_paddle_pos[0] - 10 and 
        right_paddle_pos[1] <= ball_pos[1] <= right_paddle_pos[1] + paddle_height):
        # Change the ball's horizontal velocity without a speed boost
        ball_vel[0] = -ball_vel[0]

def update_ball_position():
    # Update ball position
    ball_pos = ball_pos.astype('float64') + ball_vel

def handle_score_logic():
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

def decrease_collision_speed_boost_timer():
    # Decrease the collision_speed_boost_timer if it's active
    if collision_speed_boost_timer > 0:
        collision_speed_boost_timer -= 1

def clear_screen():
    # Clear the screen
    screen.fill((0, 0, 0))

def get_state_representation(self):
    # Gather and return the current state information
    ball_position = self.ball_pos
    left_paddle_position = self.left_paddle_pos
    right_paddle_position = self.right_paddle_pos
    # ... other relevant game state variables ...
    
    state = {
        'ball_position': ball_position,
        'left_paddle_position': left_paddle_position,
        'right_paddle_position': right_paddle_position,
    }
    
    return state

def draw_game_elements():
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

if __name__ == "__main__":
    initialize_pygame()