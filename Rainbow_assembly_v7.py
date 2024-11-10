import os
import argparse
from tianshou.utils import TensorboardLogger

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager , RainbowPolicy, C51Policy, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net

from gym_env.ma_computer_assembly_env_AEC_v7 import ComputerAssemblyMultiAgentEnv
from pettingzoo.utils import parallel_to_aec
from torch.utils.tensorboard import SummaryWriter
from atari_network import DQN


# Define the network architecture
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument("--num_atoms", type=int, default=51)

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
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="")
    # env = parallel_to_aec(env)
    env = PettingZooEnv(env)
    return env

if __name__ == "__main__":
    # Load and wrap the environment
    env = make_env()

    # Convert the env to vector format and create a collector
    envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])
    args: argparse.Namespace = get_args()

    # Observation and action space

    observation_shape = env.observation_space.shape or env.observation_space.n,
    action_shape = env.action_space.n

    # Neural networks for each agent
    # Neural networks for each agent
    # net1 = DQNNet(observation_shape, action_shape)
    # net2 = DQNNet(observation_shape, action_shape)

    net1 = Net(
        state_shape= env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        activation= nn.ReLU,
        num_atoms= args.num_atoms,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    net2 = Net(
        state_shape=env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        activation=nn.ReLU,
        num_atoms=args.num_atoms,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")



    # Optimizers
    optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    optim2 = torch.optim.Adam(net2.parameters(), lr=1e-3)


    policy1 = RandomPolicy(action_space=env.action_space)

    policy2: C51Policy = RainbowPolicy(
        model=net1,
        optim=optim1,
        discount_factor=0.9,
        action_space=env.action_space,
        num_atoms=args.num_atoms,
        estimation_step=10,
        target_update_freq=300,
    ).to('cuda')

    buffer = VectorReplayBuffer(
        100000,
        buffer_num=len(envs),
        ignore_obs_next=True,
        # save_only_last_obs=True,
        # stack_num=4,
    )
    # buffer = PrioritizedVectorReplayBuffer(
    #     args.buffer_size,
    #     buffer_num=len(envs),
    #     ignore_obs_next=True,
    #     # save_only_last_obs=False,
    #     # stack_num=4,
    #     alpha=0.5,
    #     beta=0.4,
    #     weight_norm=not False,
    # )
    policy = [policy1,policy2]
    policies = MultiAgentPolicyManager(
        policies=policy,
        env=env)




    # collector
    train_collector = Collector(policies, envs, buffer, exploration_noise=True)
    test_collector = Collector(policies, test_envs, exploration_noise=True)

    train_collector.collect(n_step=args.batch_size * 10)  # batch size * training_num



    # Policy manager


    # ======== tensorboard logging setup =========
    args: argparse.Namespace = get_args()
    log_path = os.path.join(args.logdir, "assemblygame_v7", "Rainbow")
    writer = SummaryWriter(log_path)
    # Assuming `acc_reward` is your accumulated reward for the episode and `global_step is a step counter
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Training
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "assembly", "rainbow"), exist_ok=True)
        model_save_path = os.path.join("log", "assembly", "rainbow", "policy.pth")
        torch.save(policy.policies[env.agents[1]].state_dict(), model_save_path)


    def log_episode_rewards(collector, phase, global_step, writer):
        episode_rewards = collector.buffer.rew
        if episode_rewards is not None and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar(f"{phase}/average_episode_reward", avg_reward, global_step)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1000

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        eps = 0.1
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        # policies.set_eps(eps)
        # policies.policies[env.agents[0]].set_eps(eps)
        policies.policies[env.agents[1]].set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})


    def test_fn(epoch: int, env_step: int | None) -> None:
        # policies.set_eps(args.eps_test)
        # policies.policies[env.agents[0]].set_eps(args.eps_test)
        policies.policies[env.agents[1]].set_eps(args.eps_test)


    def reward_metric(rews):
        return rews[:, 1]
    # ======== Step 5: Run the trainer =========
    result = OffpolicyTrainer(
        policy=policies,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=500,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        logger=logger,
        test_in_train=False,
        # reward_metric=reward_metric,
    ).run()
    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")