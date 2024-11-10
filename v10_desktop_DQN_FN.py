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
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor
from gym_env.computer_assembly_env_v10 import ComputerAssemblyEnv
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
        default="./data/data_DQN_v9_2k.hdf5",
    )
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.3)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




def make_env():
    env = gym.make('ComputerAssemblyEnv-v10')
    return env

if __name__ == "__main__":
    # Load and wrap the environment
    env = make_env()
    args: argparse.Namespace = get_args()

    # api_test(env, num_cycles=1000, verbose_progress=False)

    # Convert the env to vector format and create a collector
    envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])

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
        activation=torch.nn.ReLU,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    policy = DQNPolicy(model=net1,optim=optim1,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=300,
            is_double= False,
            action_space= env.action_space)


    if False:
        policy1.load_state_dict(torch.load("log/assemblygame_v9/BCQ&BCQ/policy1_DQN.pth"))



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
    log_path = os.path.join(args.logdir, "assemblygame_desktop_v10", "DQN_FN_10r")
    writer = SummaryWriter(log_path)
    # Assuming `acc_reward` is your accumulated reward for the episode and `global_step is a step counter
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Training
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "assemblygame_desktop_v10", "DQN_FN_10r"), exist_ok=True)

        model1_save_path = os.path.join("log", "assemblygame_desktop_v10", "DQN_FN_10r", "policy.pth")
        torch.save(policy.state_dict(), model1_save_path)



    def log_episode_rewards(collector, phase, global_step, writer):
        episode_rewards = collector.buffer.rew
        log_episode_part_rewards(collector, phase, global_step, writer)
        if episode_rewards is not None and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar(f"{phase}/average_episode_reward", avg_reward, global_step)


    def log_episode_part_rewards(collector, phase, global_step, writer):
        # 获取每步的 info 数据
        buffer = collector.buffer
        episode_rewards_part1 = []
        episode_rewards_part2 = []
        episode_rewards_part3 = []
        episode_rewards_part4 = []

        # Initialize temporary variables to accumulate rewards for the current episode
        current_episode_part1 = 0
        current_episode_part2 = 0
        current_episode_part3 = 0
        current_episode_part4 = 0

        # Loop through all data points in the buffer
        for i in range(len(buffer)):
            info = buffer.info[i]
            done = buffer.done[i]  # Check if the current step is the end of an episode

            # Accumulate rewards for the current episode
            current_episode_part1 += info.get('reward_agent1_part1_step', 0)
            current_episode_part2 += info.get('reward_agent1_part2_completion', 0)
            current_episode_part3 += info.get('reward_agent1_part3_failAction', 0)
            current_episode_part4 += info.get('reward_agent1_part4_fatigue', 0)

            # When an episode ends (done is True), record the accumulated rewards
            if done:
                # Append accumulated rewards for each part
                episode_rewards_part1.append(current_episode_part1)
                episode_rewards_part2.append(current_episode_part2)
                episode_rewards_part3.append(current_episode_part3)
                episode_rewards_part4.append(current_episode_part4)

                # Reset accumulation variables for the next episode
                current_episode_part1 = 0
                current_episode_part2 = 0
                current_episode_part3 = 0
                current_episode_part4 = 0

        # Calculate the average rewards across episodes
        avg_reward_part1 = np.mean(episode_rewards_part1) if episode_rewards_part1 else 0
        avg_reward_part2 = np.mean(episode_rewards_part2) if episode_rewards_part2 else 0
        avg_reward_part3 = np.mean(episode_rewards_part3) if episode_rewards_part3 else 0
        avg_reward_part4 = np.mean(episode_rewards_part4) if episode_rewards_part4 else 0

        # Log the averages to TensorBoard
        writer.add_scalar(f"{phase}/average_episode_reward_part1_step", avg_reward_part1, global_step)
        writer.add_scalar(f"{phase}/average_episode_reward_part2_completion", avg_reward_part2, global_step)
        writer.add_scalar(f"{phase}/average_episode_reward_part3_failAction", avg_reward_part3, global_step)
        writer.add_scalar(f"{phase}/average_episode_reward_part4_fatigue", avg_reward_part4, global_step)


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
