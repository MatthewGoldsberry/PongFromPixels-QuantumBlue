print("test")
import pygame
print("test2")
import numpy as np
print("test3")
import warnings
print("test4")
# Turns off warnings so I can see debugging code when it posts in terminal (DO NOT REMOVE)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import gym
print("test5")
import pong_env
print("test6")
import pong_ai

print("test7")
import ai_paddle
print("test8")

print("import complete")

def main():
    env = pong_env.PongEnv()  # Create the Pong environment

    # Initialize Pygame
    pygame.init()

    # Constants
    WIDTH, HEIGHT = 800, 400
    BALL_SPEED = 1.25
    PADDLE_SPEED = 1
    WHITE = (255, 255, 255)

    running = True

    print("Entering the game loop")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Call your AI function to choose an action
        state = env.get_state_representation()
        ai_action = pong_ai.choose_action(state)

        # Update the game state based on AI's action
        env.perform_action(ai_action)
       
        env.update_paddle_position()
        env.handle_ball_collisions()
        env.check_paddle_collisions()
        env.update_ball_position()
        env.handle_score_logic()
        env.decrease_collision_speed_boost_timer()
        env.clear_screen()
        env.draw_game_elements()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()