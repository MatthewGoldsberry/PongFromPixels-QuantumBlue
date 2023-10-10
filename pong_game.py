import pygame
import numpy as np
import ai_paddle  # Import the AI script

class PongGame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Constants
        self.WIDTH, self.HEIGHT = 800, 400
        self.BALL_SPEED = 1.25
        self.PADDLE_SPEED = 1
        self.WHITE = (255, 255, 255)
        
        # Initialize game variables
        self.ball_pos = np.array([self.WIDTH // 2, self.HEIGHT // 2])
        self.ball_vel = np.array([self.BALL_SPEED, self.BALL_SPEED])
        self.paddle_height = 100
        self.left_paddle_pos = np.array([20, self.HEIGHT // 2 - self.paddle_height // 2])
        self.right_paddle_pos = np.array([self.WIDTH - 40, self.HEIGHT // 2 - self.paddle_height // 2])
        self.paddle_vel = 0
        
        # Initialize other game state variables
        self.left_score = 0
        self.right_score = 0
        self.collision_speed_boost_timer = 0
        
        # Font and text settings
        self.font = pygame.font.Font(None, 36)
        self.text_color = (255, 255, 255)
        
        # Create the screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong")
        
        # FPS tracking variables
        self.clock = pygame.time.Clock()
        
        # Game loop
        self.running = True
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.paddle_vel = -1  # Move the paddle up
                elif event.key == pygame.K_DOWN:
                    self.paddle_vel = 1  # Move the paddle down
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    self.paddle_vel = 0  # Stop the paddle when the key is released
    
    def update(self):
        # Call clock.tick() to control the frame rate (e.g., limit to 60 FPS)
        self.clock.tick(300)
    
        # Calculate the FPS after clock.tick()
        fps = int(self.clock.get_fps())
    
        # Call the AI function and pass the height of the right paddle
        ai_paddle.ai_move(self.ball_pos, self.right_paddle_pos, self.paddle_height, self.HEIGHT, self.PADDLE_SPEED)
        
        # Update paddle positions
        self.left_paddle_pos[1] += self.paddle_vel
        if self.left_paddle_pos[1] < 0:
            self.left_paddle_pos[1] = 0
        elif self.left_paddle_pos[1] > self.HEIGHT - self.paddle_height:
            self.left_paddle_pos[1] = self.HEIGHT - self.paddle_height
    
        # Ball collisions
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.HEIGHT:
            self.ball_vel[1] = -self.ball_vel[1]
    
        # Check for collisions with the paddles before updating ball position
        if (self.ball_pos[0] - 10 <= self.left_paddle_pos[0] + 10 and 
            self.left_paddle_pos[1] <= self.ball_pos[1] <= self.left_paddle_pos[1] + self.paddle_height):
            # Change the ball's horizontal velocity without a speed boost
            self.ball_vel[0] = -self.ball_vel[0]
    
        if (self.ball_pos[0] + 10 >= self.right_paddle_pos[0] - 10 and 
            self.right_paddle_pos[1] <= self.ball_pos[1] <= self.right_paddle_pos[1] + self.paddle_height):
            # Change the ball's horizontal velocity without a speed boost
            self.ball_vel[0] = -self.ball_vel[0]
            
        # Update ball position
        self.ball_pos = self.ball_pos.astype('float64') + self.ball_vel
    
        # Ball out of bounds - handle scoring logic here
        if self.ball_pos[0] <= 0:
            # Right paddle scores a point
            self.right_score += 1
            self.ball_pos = np.array([self.WIDTH // 2, self.HEIGHT // 2])
            self.ball_vel = np.array([self.BALL_SPEED, self.BALL_SPEED])
        elif self.ball_pos[0] >= self.WIDTH:
            # Left paddle scores a point
            self.left_score += 1
            self.ball_pos = np.array([self.WIDTH // 2, self.HEIGHT // 2])
            self.ball_vel = np.array([-self.BALL_SPEED, self.BALL_SPEED])
    
        # Decrease the collision_speed_boost_timer if it's active
        if self.collision_speed_boost_timer > 0:
            self.collision_speed_boost_timer -= 1
    
    def draw(self):
        # Clear the screen
        self.screen.fill((0, 0, 0))
    
        # Create text surfaces for left and right scores
        left_score_text = self.font.render(f"Left Score: {self.left_score}", True, self.text_color)
        right_score_text = self.font.render(f"Right Score: {self.right_score}", True, self.text_color)
    
        # Blit (draw) the score text surfaces onto the screen at specific positions
        self.screen.blit(left_score_text, (10, 10))  # Adjust the position as needed
        self.screen.blit(right_score_text, (self.WIDTH - 200, 10))  # Adjust the position as needed
    
        # Draw paddles and ball
        pygame.draw.rect(self.screen, self.WHITE, (self.left_paddle_pos[0], self.left_paddle_pos[1], 20, self.paddle_height))
        pygame.draw.rect(self.screen, self.WHITE, (self.right_paddle_pos[0], self.right_paddle_pos[1], 20, self.paddle_height))
        pygame.draw.ellipse(self.screen, self.WHITE, (self.ball_pos[0] - 10, self.ball_pos[1] - 10, 20, 20))
    
        # Update the display
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
        
        pygame.quit()

# Create an instance of the PongGame class and run the game
if __name__ == "__main__":
    game = PongGame()
    game.run()