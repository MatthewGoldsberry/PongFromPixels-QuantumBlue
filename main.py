print("test")
import pygame
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import gym
import ai_paddle
from pong_env import PongEnv, get_state_representation, initialize_game, handle_events, update_paddle_position, handle_ball_collisions, check_paddle_collisions, update_ball_position, handle_score_logic, decrease_collision_speed_boost_timer, clear_screen, draw_game_elements
import pong_ai


# Initialize Pygame
pygame.init()

# Turns off warnings so I can see debugging code when it posts in terminal (DO NOT REMOVE)
gym.logger.set_level(ERROR)

def main():
    # Constants and initialization done in PongEnv
    running = True
    initialize_game(WIDTH, HEIGHT)
    
    while running:
        

        ai_paddle.ai_move(ball_pos, right_paddle_pos, paddle_height, HEIGHT, PADDLE_SPEED)

        # Call your AI function to choose an action
        ai_action = pong_ai.choose_action(get_state_representation())
        print(ai_action)
        
        # Update the game state based on AI's action
        perform_action(ai_action)
        print("Calling preform_action function")

        handle_events(ai_action)
        update_paddle_position()
        handle_ball_collisions()
        check_paddle_collisions()
        update_ball_position()
        handle_score_logic()
        decrease_collision_speed_boost_timer()
        clear_screen()
        draw_game_elements()
        
        pygame.display.flip()

if __name__ == "__main__":
    WIDTH, HEIGHT, BALL_SPEED, PADDLE_SPEED, ball_pos, ball_vel, paddle_height, left_paddle_pos, right_paddle_pos, paddle_vel = env.reset()
    clock = pygame.time.Clock()
    main()

# Quit Pygame
pygame.quit()