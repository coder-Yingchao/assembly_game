import os
import argparse
from tianshou.utils import TensorboardLogger

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager, DiscreteBCQPolicy, ImitationPolicy
from tianshou.trainer import  OffpolicyTrainer
from tianshou.utils.net.common import Net, ActorCritic,  Transformer
from tianshou.utils.net.discrete import Actor
from gym_env.computer_assembly_env_v11 import ComputerAssemblyEnv
from gym_env.computer_assembly_env_pump_v11 import ComputerAssemblyEnv
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--update-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=1)
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
        default="./data/data_DQN_v11_2k.hdf5",
    )
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.3)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




def make_env():
    env = gym.make('ComputerAssemblyEnv-pump-v11')
    return env

if __name__ == "__main__":
    # Load and wrap the environment
    env = make_env()
    args: argparse.Namespace = get_args()

    # api_test(env, num_cycles=1000, verbose_progress=False)

    # Convert the env to vector format and create a collector
    envs = DummyVectorEnv([lambda: make_env() for _ in range(5)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(2)])

    # Observation and action space

    observation_shape = env.observation_space['observation'].shape or env.observation_space['observation'].n,
    action_shape = env.action_space.n

    # Neural networks for each agent
    # Neural networks for each agent
    # net1 = DQNNet(observation_shape, action_shape)
    # net2 = DQNNet(observation_shape, action_shape)
    net1 = Transformer(
        input_dim= observation_shape,
        hidden_dim= 128,
        num_heads= 4,
        num_layers= 1,   # 1 or 2 are both fine.
        output_dim= action_shape
    )
    optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    policy = DQNPolicy(model=net1,optim=optim1,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=300,
            action_space= env.action_space)


    if False:
        policy1.load_state_dict(torch.load("log/assemblygame_v11/BCQ&BCQ/policy1_DQN.pth"))



    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        envs,
        VectorReplayBuffer(20_000, len(envs)),
        exploration_noise=True,
    )
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "assemblygame_pump_random_5_v11", "DQN_Transform_50")
    writer = SummaryWriter(log_path)
    # Assuming `acc_reward` is your accumulated reward for the episode and `global_step is a step counter
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Training
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "assemblygame_pump_random_5_v11", "DQN_Transform_50"), exist_ok=True)

        model1_save_path = os.path.join("log", "assemblygame_pump_random_5_v11", "DQN_Transform_50", "policy.pth")
        torch.save(policy.state_dict(), model1_save_path)



    def log_episode_rewards(collector, phase, global_step, writer):
        episode_rewards = collector.buffer.rew
        if episode_rewards is not None and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar(f"{phase}/average_episode_reward", avg_reward, global_step)
    def train_fn(epoch, env_step):
        eps = max(0.1, 1 - epoch * 0.003)  # Example of linearly decreasing epsilon

        # policies.policies[env.agents[0]].set_eps(eps)
        policy.set_eps(eps)
        log_episode_rewards(train_collector, "train", epoch, writer)



    def test_fn(epoch, env_step):
        # policies.policies[env.agents[0]].set_eps(0.05)
        policy.set_eps(0.05)
        log_episode_rewards(train_collector, "test", epoch, writer)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1000

    def reward_metric(rews):
        return rews
    # ======== Step 5: Run the trainer =========
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1000,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    ).run()
    print(f"\n==========Result==========\n{result}")
