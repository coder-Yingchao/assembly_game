# ROS_compute_action.py
import sys
import json
import torch
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
from tianshou.env.pettingzoo_env import PettingZooEnv
from gym_env.ma_computer_assembly_env_AEC_v9 import ComputerAssemblyMultiAgentEnv

def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="human")
    env = PettingZooEnv(env)
    return env

def compute_action(observation, info=None, state=None):
    env = make_env()
    net1 = Net(
        state_shape=env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        activation=torch.nn.ReLU,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    policy = DQNPolicy(model=net1, optim=optim1,
                       discount_factor=0.9,
                       estimation_step=3,
                       target_update_freq=300,
                       is_double=False,
                       action_space=env.action_space)
    policy.load_state_dict(torch.load("log/assemblygame_desktop_test_v9/DQN_FN/policy.pth"))
    return policy.compute_action(observation, info, state)


if __name__ == "__main__":
    # Read and parse observation from command-line arguments
    observation_json = sys.argv[1]
    observation = json.loads(observation_json)

    # Compute action
    action = compute_action(observation)

    # Output the action for the main script to capture
    print(action)
