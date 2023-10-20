import pygame
import numpy as np
import ai_paddle  # Import the AI script
import random

# Constants
WIDTH, HEIGHT = 800, 400
BALL_SPEED = 1.25
PADDLE_SPEED = 1
WHITE = (255, 255, 255)
left_score = 0
right_score = 0

# Initialize game variables
ball_pos = np.array([WIDTH // 2, HEIGHT // 2])
ball_vel = np.array([BALL_SPEED, BALL_SPEED])
paddle_height = 100
left_paddle_pos = np.array([20, HEIGHT // 2 - paddle_height // 2])
right_paddle_pos = np.array([WIDTH - 40, HEIGHT // 2 - paddle_height // 2])
paddle_vel = 0

# Initialize Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
max_episodes = 1000  # Maximum number of episodes
episode = 0

# Define the action space
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_STAY = 2
action_space = [ACTION_UP, ACTION_DOWN, ACTION_STAY]

# Create the Q-table and initialize with zeros
# q_table = {}
# print(q_table)
# state_space = [(ball_x, ball_y, left_paddle_y, right_paddle_y) for ball_x in range(WIDTH) for ball_y in range(HEIGHT) for left_paddle_y in range(HEIGHT) for right_paddle_y in range(HEIGHT)]
# for state in state_space:
#     for action in action_space:
#         q_table[(state, action)] = 0.0
#         print(q_table)

def get_state(ball_pos, left_paddle_pos):
    difference = ball_pos[1] - left_paddle_pos[1]

    if difference <= -paddle_height:
        return 0
    elif difference <= -paddle_height // 2:
        return 1
    elif difference < 0:
        return 2
    elif difference == 0:
        return 3
    elif difference < paddle_height // 2:
        return 4
    elif difference < paddle_height:
        return 5
    elif difference < paddle_height * 3 / 2:
        return 6
    else:
        return 7

def select_best_action(q_table, state):
    print("select best action called")
    best_action = None
    best_q_value = float('-inf')

    for action in action_space:
        q_value = q_table.get((state, action), 0.0)
        if q_value > best_q_value:
            best_q_value = q_value
            best_action = action

    return best_action

def get_reward(current_state, action, new_state):
    print("get_reward called")
    # Extract relevant variables from the state
    ball_x, ball_y, left_paddle_y, right_paddle_y = current_state

    # Initialize the reward
    reward = 0

    # If the AI-controlled right paddle successfully hits the ball
    if action == ACTION_STAY and ball_x > WIDTH / 2:
        reward = 1

    # If the AI-controlled right paddle fails to hit the ball
    if action == ACTION_STAY and ball_x < WIDTH / 2:
        reward = -1

    # If the ball passes the AI and scores a point for the player-controlled left paddle
    if ball_x > WIDTH / 2 and new_state[0] < WIDTH / 2:
        reward = -10

    # If the ball passes the AI and scores a point for the AI-controlled right paddle
    if ball_x < WIDTH / 2 and new_state[0] > WIDTH / 2:
        reward = 10

    return reward


def take_action(action, current_state):
    print("take action called")
    if action == ACTION_UP:
        left_paddle_pos[1] -= PADDLE_SPEED
    elif action == ACTION_DOWN:
        left_paddle_pos[1] += PADDLE_SPEED

    # Ensure the left paddle doesn't go out of bounds
    if left_paddle_pos[1] < 0:
        left_paddle_pos[1] = 0
    elif left_paddle_pos[1] > HEIGHT - paddle_height:
        left_paddle_pos[1] = HEIGHT - paddle_height

    new_state = get_state(ball_pos, left_paddle_pos)

    return new_state

pygame.init()

# Game loop
running = True
print("Running =", running)
# Initialize Pygame


# Font and text settings
self.font = pygame.font.Font(None, 36)  # You can adjust the font size and style
text_color = (255, 255, 255)  # White text color

# Create the screen
self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

while running:
    print("count")
    count += 1
    print("in game loop")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                paddle_vel = -1  # Move the paddle up
            elif event.key == pygame.K_DOWN:
                paddle_vel = 1  # Move the paddle down
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                paddle_vel = 0  # Stop the paddle when the key is released
        elif count == 10:
            print("count reached")
            break

    # FPS tracking variables
    clock = pygame.time.Clock()
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

    current_state = get_state(ball_pos, ball_vel, left_paddle_pos, right_paddle_pos)
    if random.random() < epsilon:
        # Exploration: Choose a random action
        chosen_action = random.choice(action_space)
    else:
        # Exploitation: Choose the action with the highest Q-value for the current state
        chosen_action = select_best_action(q_table, current_state)

    # Access the current Q-value for the chosen state-action pair
    q_value = q_table[(current_state, chosen_action)]

    # Implement the take_action function with your game-specific logic
    next_state, reward = take_action(chosen_action, current_state)

    # Calculate the maximum Q-value for the next state
    max_q_value_next_state = max(q_table[(next_state, action)] for action in action_space)

    # Update the Q-value using the Q-learning update rule
    q_table[(current_state, chosen_action)] += alpha * (reward + gamma * max_q_value_next_state - q_value)
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

    print("draw")
    # Blit (draw) the score text surfaces onto the screen at specific positions
    screen.blit(left_score_text, (10, 10))  # Adjust the position as needed
    screen.blit(right_score_text, (WIDTH - 200, 10))  # Adjust the position as needed
    screen.blit(fps_text, (340, 10))  # Adjust the position as needed

    # Draw paddles and ball
    pygame.draw.rect(screen, WHITE, pygame.Rect(left_paddle_pos[0], left_paddle_pos[1], 20, paddle_height))
    pygame.draw.rect(screen, WHITE, pygame.Rect(right_paddle_pos[0], right_paddle_pos[1], 20, paddle_height))
    pygame.draw.ellipse(screen, WHITE, (ball_pos[0] - 10, ball_pos[1] - 10, 20, 20))


    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.font.quit()
pygame.quit()
