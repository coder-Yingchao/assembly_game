import os
import argparse
from tianshou.utils import TensorboardLogger

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import  MultiAgentPolicyManager, ICMPolicy, PPOPolicy, DQNPolicy, RandomPolicy
from tianshou.utils.net.common import Net
from pettingzoo.utils import parallel_to_aec
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from torch.distributions import Categorical, Distribution
from tianshou.trainer import OnpolicyTrainer
from gym_env.computer_assembly_env_v7 import ComputerAssemblyEnv



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
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
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
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--repeat-per-collect", type=int, default=4)


    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




def make_env():
    env = gym.make('ComputerAssemblyEnv-v7')
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
        state_shape= observation_shape,
        action_shape= action_shape,
        hidden_sizes=[128, 128, 128, 128],
        device= args.device,
    ).to(args.device)


    actor1 = Actor(net1, action_shape, device=args.device, softmax_output=False)
    critic1 = Critic(net1, device=args.device)

    net2 = Net(
        state_shape= observation_shape,
        action_shape= action_shape,
        hidden_sizes=[128, 128, 128, 128],
        device= args.device,
    ).to(args.device)


    # Optimizers
    optim1 = torch.optim.Adam(ActorCritic(actor1, critic1).parameters(), lr=args.lr, eps=1e-5)
    optim2 = torch.optim.Adam(net2.parameters(), lr=1e-3, eps= 1e-5)
    def dist(logits: torch.Tensor) -> Distribution:
        return Categorical(logits=logits)
    lr_scheduler = None

    policy = DQNPolicy(model=net2,optim=optim2,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=300,
            action_space= env.action_space)

    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(envs),
        ignore_obs_next=True,
        save_only_last_obs=False
    )


    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        envs,
        buffer,
        exploration_noise=True,
    )

    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "assemblygame_v7", "PPO")
    writer = SummaryWriter(log_path)
    # Assuming `acc_reward` is your accumulated reward for the episode and `global_step is a step counter
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Training
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "assemblygame_v7", "PPO"), exist_ok=True)
        model2_save_path = os.path.join("log", "assemblygame_v7", "PPO", "policy.pth")
        # torch.save(policy.policies[env.agents[0]].state_dict(), model1_save_path)
        torch.save(policy.state_dict(), model2_save_path)

    def log_episode_rewards(collector, phase, global_step, writer):
        episode_rewards = collector.buffer.rew
        if episode_rewards is not None and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar(f"{phase}/average_episode_reward", avg_reward, global_step)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1500

    def train_fn(epoch, env_step):
        eps = max(0.1, 1 - epoch * 0.003)  # Example of linearly decreasing epsilon

        # policies.policies[env.agents[0]].set_eps(eps)
        policy.set_eps(eps)
        log_episode_rewards(train_collector, "train", epoch, writer)



    def test_fn(epoch, env_step):
        # policies.policies[env.agents[0]].set_eps(0.05)
        policy.set_eps(0.05)
        log_episode_rewards(train_collector, "test", epoch, writer)


    def reward_metric(rews):
        return rews[:, 1]
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== Step 5: Run the trainer =========
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=False
    ).run()
    print(f"\n==========Result==========\n{result}")
