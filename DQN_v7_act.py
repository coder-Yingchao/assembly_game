import os
import argparse
from tianshou.utils import TensorboardLogger

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager, DiscreteBCQPolicy, ImitationPolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils.net.common import Net, ActorCritic, Recurrent
from tianshou.utils.net.discrete import Actor
import time
import pygame
import sys
from tianshou.env.pettingzoo_env import PettingZooEnv
from gym_env.ma_computer_assembly_env_AEC_v7 import ComputerAssemblyMultiAgentEnv


# Define the network architecture
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--update-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda"
    )
    parser.add_argument(
        "--load-buffer-name",
        type=str,
        default="./data/data_20_epo_Penaltyeverystep.hdf5",
    )
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.3)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="human")
    env = PettingZooEnv(env)
    return env

if __name__ == "__main__":
    # Load and wrap the environment
    env = make_env()
    args: argparse.Namespace = get_args()

    # Convert the env to vector format and create a collector
    envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])

    # Observation and action space

    observation_shape = env.observation_space.shape or env.observation_space.n,
    action_shape = env.action_space.n

    net1 = Net(
        state_shape= env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        activation=torch.nn.ReLU,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    policy = DQNPolicy(model=net1,optim=optim1,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=300,
            action_space= env.action_space)

    if True:
        policy.load_state_dict(torch.load("log/assemblygame_v7/DQN_Relu/policy.pth"))
        print("Loaded agent from: pth")

    pygame.init()
    observation, info = env.reset()

    print(observation)

    screen = pygame.display.set_mode((1600, 900))
    pygame.display.set_caption('Gym Environment Interaction')

    # Define action mappings for both agents
    agent1_actions = {
        pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
        pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7,
        pygame.K_8: 8, pygame.K_9: 9
    }
    agent2_actions = {
        pygame.K_KP0: 0, pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3,
        pygame.K_KP4: 4, pygame.K_KP5: 5, pygame.K_KP6: 6, pygame.K_KP7: 7,
        pygame.K_KP8: 8, pygame.K_KP9: 9
    }

    # Initialize current actions with a neutral value within your action space if necessary
    current_action_agent1 = -1
    current_action_agent2 = -1
    # Game loop
    running = True
    init_obs = observation

    while running:
        action_performed = False
        # Event loop
        while not action_performed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    action_performed = True  # End this inner loop if we're quitting
                elif event.type == pygame.KEYDOWN:
                    if event.key in agent1_actions:
                        current_action_agent1 = agent1_actions[event.key]
                        action_performed = True  # Break the inner loop if action is detected
        time.sleep(1)
        env.render()

        # Environment interaction logic
        # # Simulate environment step with the current actions of both agents
        # action = {'agent_1':(current_action_agent1),
        #           'agent_2': (current_action_agent2)}
        observation, reward, term, truncated, info = env.step(current_action_agent1 + 1)
        info = None  # Additional info if available
        state = None  # State if your policy is recurrent
        observation = observation['obs']
        act = policy.compute_action(observation, info, state)
        print('act:',act)

        observation2, reward2, term2, truncated2, info2 = env.step(act)
        # print(reward2)

        # Record the step
        # record_step(init_obs, current_action_agent1 + 1, observation, reward, term, truncated)
        # record_step(observation, current_action_agent2 + 1, observation2, reward2, term2, truncated2)

        if term2 or truncated2:
            observation = env.reset()  # Reset the environment for the next episode
            data_records = []
            # save_buffer_to_hdf5(filepath)  # Save recorded data at the end of each episode

        current_action_agent2 = -1
        current_action_agent1 = -1
        init_obs = observation2

    # Cleanup
    pygame.quit()
    env.close()
    sys.exit()


