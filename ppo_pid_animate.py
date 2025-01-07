import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import pygame

# Define a custom environment for a line-following car
class LineFollowingCarEnv(gym.Env):
    def __init__(self):
        # Initialize the base class for a custom Gymnasium environment
        super(LineFollowingCarEnv, self).__init__()

        # Define the observation space
        # Observation includes:
        # - position_error: how far the car is from the line (-10 to 10)
        # - angle_to_line: the car's angle relative to the line (-π to π)
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -np.pi]),  # Minimum values
            high=np.array([10.0, np.pi]),  # Maximum values
            dtype=np.float32
        )

        # Define the action space
        # Actions are the PID gains [Kp, Ki, Kd] (Proportional, Integral, Derivative)
        # These values range from 0 to 10
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),  # Minimum gains
            high=np.array([10.0, 10.0, 10.0]),  # Maximum gains
            dtype=np.float32
        )
        
        # Initialize Pygame for visualization
        pygame.init()
        self.width, self.height = 800, 400  # Window dimensions
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Line Following Car")  # Title of the Pygame window
        self.clock = pygame.time.Clock()  # Control the frame rate

        # Define colors for visualization
        self.white = (255, 255, 255)  # Background color
        self.black = (0, 0, 0)        # Text color (unused here)
        self.red = (255, 0, 0)        # Line color
        self.blue = (0, 0, 255)       # Car color

        # Define car properties
        self.car_pos_x = self.width // 2  # Car's horizontal position
        self.car_pos_y = self.height // 2  # Car's vertical position
        self.car_width, self.car_height = 40, 20  # Car dimensions

        # Reset the environment to initialize all variables
        self.reset()

    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility (optional)
        if seed is not None:
            np.random.seed(seed)

        # Initialize the car's state
        self.position_error = np.random.uniform(-2.0, 2.0)  # Random initial offset from the line
        self.angle_to_line = np.random.uniform(-np.pi / 4, np.pi / 4)  # Random initial angle
        self.time_step = 0  # Reset the time step counter
        self.total_error = 0.0  # Reset the accumulated error for the integral term

        # Start the car at the left edge of the window
        self.car_pos_x = 0  

        # Return the initial observation (state) and an empty info dictionary
        return np.array([self.position_error, self.angle_to_line], dtype=np.float32), {}

    def step(self, action):
        # Unpack the PID gains (actions taken by the agent)
        Kp, Ki, Kd = action

        # Compute the PID correction
        correction = (
            Kp * self.position_error +                     # Proportional term
            Ki * self.total_error +                       # Integral term (sum of past errors)
            Kd * (self.position_error - getattr(self, "prev_position_error", 0))  # Derivative term
        )

        # Update the car's state based on the correction
        self.prev_position_error = self.position_error  # Store the previous position error
        self.position_error -= correction * np.cos(self.angle_to_line)  # Adjust position error
        self.angle_to_line -= correction * 0.1  # Adjust the angle slightly (simplified physics)

        # Move the car forward (right) and adjust its vertical position based on the error
        self.car_pos_x += 5  # Move horizontally by 5 units per step
        self.car_pos_y = self.height // 2 + int(self.position_error * 20)  # Adjust vertical position

        # Clamp the position error and angle to their defined limits
        self.position_error = np.clip(self.position_error, -10, 10)
        self.angle_to_line = np.clip(self.angle_to_line, -np.pi, np.pi)

        # Compute the reward
        # Reward is higher (closer to 0) when the car stays near the line with minimal corrections
        reward = -abs(self.position_error) - 0.01 * abs(correction)

        # Accumulate the position error for the integral term
        self.total_error += self.position_error

        # Determine if the episode is done
        # - Episode ends if the car runs for 200 timesteps or moves too far off the line
        self.time_step += 1
        done = self.time_step >= 200 or abs(self.position_error) > 10

        # Return the updated state, reward, and whether the episode is done
        return (
            np.array([self.position_error, self.angle_to_line], dtype=np.float32),
            reward,
            done,
            False,
            {}
        )
            
    def render(self):
        # Handle Pygame events (e.g., close the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If the close button is clicked
                self.close()
                exit()

        # Clear the screen with a white background
        self.screen.fill(self.white)

        # Draw the red line (the target line the car should follow)
        pygame.draw.line(
            self.screen,
            self.red,
            (0, self.height // 2),  # Start point of the line
            (self.width, self.height // 2),  # End point of the line
            2,  # Thickness of the line
        )

        # Draw the car as a blue rectangle
        pygame.draw.rect(
            self.screen,
            self.blue,
            (self.car_pos_x - self.car_width // 2, self.car_pos_y - self.car_height // 2, self.car_width, self.car_height),
        )

        # Update the display to reflect the changes
        pygame.display.flip()

        # Limit the frame rate to 30 FPS
        self.clock.tick(30)

    def close(self):
        # Quit Pygame to clean up resources
        pygame.quit()


def train():

    # Create the environment
    env = LineFollowingCarEnv()

    # Train PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("ppo_line_following_car")
    print("Model saved.")
    
def evaluate():
    env = LineFollowingCarEnv()
    model = PPO.load("ppo_pid_car")  # Load the trained model

    obs, _ = env.reset()
    done, truncated = False, False
    total_reward = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)  # Predict action using PPO
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        env.render()  # Render the environment
        
    print(f"Total reward: {total_reward}")
    env.close()
    
if __name__ == "__main__":
    evaluate()
