import pygame
import numpy as np
import math

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

class PongGame:
    def __init__(self):
        self.init_pygame()
        self.reset()

    def init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Custom Pong Environment')
        self.font = pygame.font.Font(None, 18)  # Font for the scoreboard

    def reset(self):
        # Reset paddle positions
        self.paddle_position = HEIGHT / 2
        self.opponent_paddle_position = HEIGHT / 2

        # Reset ball position
        self.ball_position = {
            'x': WIDTH / 2,
            'y': HEIGHT / 2
        }

        # Reset ball speed with random direction
        self.ball_speed_x = BALL_SPEED * (np.random.randint(2) * 2 - 1)
        self.ball_speed_y = BALL_SPEED * (np.random.randint(2) * 2 - 1)

        # Reset game score
        global left_score, right_score
        left_score = 0
        right_score = 0

    def update_display(self):
        self.screen.fill((0, 0, 0))  # Fill the screen with black

        # Draw paddles
        pygame.draw.rect(self.screen, WHITE,
                         pygame.Rect(20, self.paddle_position - 50, 20, 100))
        pygame.draw.rect(self.screen, WHITE,
                         pygame.Rect(WIDTH - 40, self.opponent_paddle_position - 50, 20, 100))

        # Draw ball
        pygame.draw.ellipse(self.screen, WHITE,
                            (self.ball_position['x'] - 10, self.ball_position['y'] - 10, 20, 20))

        # Draw scores
        left_score_text = self.font.render(f"Left Score: {left_score}", True, WHITE)
        right_score_text = self.font.render(f"Right Score: {right_score}", True, WHITE)
        self.screen.blit(left_score_text, (10, 10))
        self.screen.blit(right_score_text, (WIDTH - 200, 10))

        pygame.display.flip()  # Update the display

    def take_action(self, action):
        # Apply the action to the environment and calculate new state and reward
        # Update the game state based on the chosen action
        self.update_paddle_position(action, False)
        self.update_ball_position()

    def update_paddle_position(self, action, is_opponent):
        # Paddle movement logic
        if is_opponent:
            if action == 0:  # Move paddle up
                self.opponent_paddle_position = max(0, self.opponent_paddle_position - PADDLE_SPEED)
            elif action == 1:  # Move paddle down
                self.opponent_paddle_position = min(HEIGHT - 100, self.opponent_paddle_position + PADDLE_SPEED)
        else:
            if action == 0:  # Move paddle up
                self.paddle_position = max(0, self.paddle_position - PADDLE_SPEED)
            elif action == 1:  # Move paddle down
                self.paddle_position = min(HEIGHT - 100, self.paddle_position + PADDLE_SPEED)

    def update_ball_position(self):
        # Update the ball's position based on its speed
        self.ball_position['x'] += self.ball_speed_x
        self.ball_position['y'] += self.ball_speed_y

        # Ball collisions with top and bottom walls
        if self.ball_position['y'] <= 0 or self.ball_position['y'] >= HEIGHT:
            self.ball_speed_y = -self.ball_speed_y

        # Check for collisions with the paddles before updating ball position
        if (self.ball_position['x'] <= 30 and
            self.paddle_position <= self.ball_position['y'] <= self.paddle_position + 100):
            # Change the ball's horizontal velocity without a speed boost
            self.ball_speed_x = -self.ball_speed_x
        elif (self.ball_position['x'] >= WIDTH - 40 and
              self.opponent_paddle_position <= self.ball_position['y'] <= self.opponent_paddle_position + 100):
            # Change the ball's horizontal velocity without a speed boost
            self.ball_speed_x = -self.ball_speed_x

        # Ball out of bounds - handle scoring logic here
        if self.ball_position['x'] <= 0:
            # Right paddle scores a point
            global right_score
            right_score += 1
            self.reset_ball()
        elif self.ball_position['x'] >= WIDTH:
            # Left paddle scores a point
            global left_score
            left_score += 1
            self.reset_ball()

    def reset_ball(self):
        # Reset ball position to the center
        self.ball_position = {'x': WIDTH / 2, 'y': HEIGHT / 2}
        # Reset ball speed with random direction
        self.ball_speed_x = BALL_SPEED * (np.random.randint(2) * 2 - 1)
        self.ball_speed_y = BALL_SPEED * (np.random.randint(2) * 2 - 1)