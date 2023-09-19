import pygame
import numpy as np

def ai_move(ball_pos, right_paddle_pos, right_paddle_height, HEIGHT, PADDLE_SPEED):
    # Implement AI logic to control the right paddle's movement
    # Use the provided right_paddle_height to adjust AI behavior

    # Calculate the desired y-coordinate for the right paddle
    desired_y = ball_pos[1] - right_paddle_height // 2

    # Adjust the right paddle's position to move towards the desired y-coordinate
    if right_paddle_pos[1] < desired_y:
        right_paddle_pos[1] += PADDLE_SPEED
    elif right_paddle_pos[1] > desired_y:
        right_paddle_pos[1] -= PADDLE_SPEED

    # Ensure the paddle stays within the screen bounds
    if right_paddle_pos[1] < 0:
        right_paddle_pos[1] = 0
    elif right_paddle_pos[1] > HEIGHT - right_paddle_height:
        right_paddle_pos[1] = HEIGHT - right_paddle_height
