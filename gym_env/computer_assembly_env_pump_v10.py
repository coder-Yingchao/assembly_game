import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from game.computer_assembly_game_pump_v8 import ComputerAssemblyGame
from gymnasium.envs.registration import register
# gym env, which has only one agent.
register(
    id='ComputerAssemblyEnv-pump-v10',  # Use an environment ID with a version number
    entry_point='gym_env.computer_assembly_env_pump_v10:ComputerAssemblyEnv',  # Module path : ClassName
)

max_steps = 3000  # Define a maximum number of steps
current_step = 0  # Track the current step of the episode

class ComputerAssemblyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ComputerAssemblyEnv, self).__init__()
        self.game = ComputerAssemblyGame()
        self.action_space =  spaces.Discrete(27, start=0)  # Each hand can perform 9 actions
        n_components = 26  # Or however many components you have
        multi_discrete_obs_space = [4] * n_components
        self.observation_space =  spaces.MultiDiscrete(np.array(multi_discrete_obs_space))
        self.human_lazy_hard = random.randint(0,1)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Reset the game to its initial state
        self.game.reset_game(seed)  # Assuming reset_game doesn't need a seed. If it does, pass seed here.
        observation = self._get_observation()  # Get the initial game state as observation
        info = {}
        self.human_lazy_hard = random.randint(0,1)

        return observation, info



    def step(self, action):
        # Increment the step counter
        global current_step, max_steps
        current_step += 1

        # Perform the action and get the reward
        rewards, terminated = self.game.move_with_random_action( -1, action, self.human_lazy_hard)
        reward = rewards[0]
        observation = self._get_observation()
        done = terminated  # This could be set based on the game's termination condition
        # Check if the episode should be truncated
        truncated = current_step >= max_steps
        if truncated:
            done = True  # Ensure "done" is also True when truncating

        # Reset the step counter if the episode is done
        if done:
            current_step = 0
            # print(f'{reward}')
        info = {}
        return observation, reward, done, truncated, info

    def render(self):
        self.game.render()


    def close(self):
        pygame.quit()

    def _get_observation(self):
        # Initialize an empty list to hold observation data
        obs = []
        # Loop through each component and append its position and state to the observation list
        for component_name, component_data in self.game.states.items():
            # obs.append(int(component_data['position'][0]))  # x position, cast to int
            # obs.append(int(component_data['position'][1]))  # y position, cast to int
            obs.append(int(component_data['state'])-1)  # state, cast to int
        # if self.game.hand2_waiting_for_handover:
        #     obs.append(1)
        # else:
        #     obs.append(0)

        # Convert the observation list to a NumPy array with dtype=np.int32
        observation = np.array(obs, dtype=np.int32)

        return observation

    def seed(self, seed=None):
        # Optional: Set seed
        pass
