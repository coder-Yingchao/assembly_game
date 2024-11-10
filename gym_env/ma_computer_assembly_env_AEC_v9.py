import time
from typing import Any
import functools
import numpy as np
from pettingzoo import AECEnv

from gymnasium import  spaces
from pettingzoo.utils.env import ObsType

from game.computer_assembly_game_v7 import ComputerAssemblyGame
from pettingzoo.utils import  wrappers, agent_selector
import gymnasium as gym
import  pygame
# update the reward by considering the distance between parts and case
def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """

    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env    """

    env = ComputerAssemblyMultiAgentEnv(render_mode=render_mode)
    return env


class ComputerAssemblyMultiAgentEnv(AECEnv):
    metadata = {
        "name": "computer_assembly_multi_agent_env_pettingzoo",
    }

    def __init__(self,render_mode=None):
        super().__init__()  # Ensure proper initialization of base classes
        self.game = ComputerAssemblyGame()
        self.agents = ['agent_1', 'agent_2']  #1 : robot and 2 : human
        self.possible_agents = self.agents[:]
        self._action_spaces = {agent: spaces.Discrete(9, start=0)  for agent in self.agents}
        # Define observation space similarly, customized to your environment's needs
        n_components = 8
        multi_discrete_obs_space = [4] * n_components
        self._observation_spaces = {
            agent: spaces.MultiDiscrete(np.array(multi_discrete_obs_space) ) for agent in self.agents
        }
        self.max_steps = 3000
        self.render_mode = render_mode
        self.last_step_return =[]
        self.screen = 1
        self._cumulative_rewards = 0
        self.state  = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        n_components = 8
        multi_discrete_obs_space = [4] * n_components
        return spaces.MultiDiscrete(np.array(multi_discrete_obs_space))
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(9, start=0)

    def render(self):
        # time.sleep(0.1)
        # print('render')
        if self.render_mode== 'human':

            self.game.render()

    def reset(self, seed=None, options=None):
        # print(self._cumulative_rewards)
        # print(self.state)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        # self.agent_selection = self._agent_selector.next()
        self.game.reset_game(1)


    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # if self.state[self.agents[0]] is None:
            #     self.state[self.agents[0]] = 0
            rewards, terminated = self.game.move_hands_simultaneously(self.state[self.agents[0]], self.state[self.agents[1]])
            self.rewards[self.agents[0]] = rewards[0]
            self.rewards[self.agents[1]] = rewards[1]

            # print(self.state[self.agents[0]]-1, self.state[self.agents[1]]-1)


            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= self.max_steps for agent in self.agents
            }
            self.terminations ={
                agent: terminated for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.observe(i)
            self._accumulate_rewards()
            # print(self.rewards)
            # print(self.state)
        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards


        if self.render_mode == "human":
            self.render()


    def observe(self,agent):
        # Initialize an empty list to hold observation data
        obs = []

        # Loop through each component and append its position and state to the observation list
        for component_name, component_data in self.game.states.items():
            # obs.append(int(component_data['position'][0])/100)  # x position, cast to int
            # obs.append(int(component_data['position'][1])/100)  # y position, cast to int
            obs.append(int(component_data['state']-1))  # state, cast to int

        # if self.game.action_hand1 is None:
        #     obs.append(0)
        # else:
        #     obs.append(self.game.action_hand1+1)
        # obs.append(self.game.second_hand_x/100)
        # obs.append(self.game.second_hand_y/100)
        # if self.game.action_hand2 is None:
        #     obs.append(0)
        # else:
        #     obs.append(self.game.action_hand2+1)
        # if self.game.hand2_waiting_for_handover:
        #     obs.append(1)
        # else:
        #     obs.append(0)

        # Convert the observation list to a NumPy array with dtype=np.int32
        observation = np.array(obs, dtype=np.int32)

        return observation




    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None



