import os.path
import time

import pygame
import sys
from tianshou.env.pettingzoo_env import PettingZooEnv
from gym_env.ma_computer_assembly_env_AEC_v7 import ComputerAssemblyMultiAgentEnv
from tianshou.data import ReplayBuffer, Batch, VectorReplayBuffer
def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="human")
    env = PettingZooEnv(env)
    return env

pygame.init()
env = make_env()
observation,info = env.reset()

print(observation)

screen = pygame.display.set_mode((1500, 1000))
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


# Initialize buffer
buffer_size = 20000  # Adjust this size according to your needs
num_agents = 1

# buffer = VectorReplayBuffer(buffer_size, num_agents)
buffer = ReplayBuffer(buffer_size)
filepath = 'data/data_20_v7.hdf5'
if os.path.exists(filepath):
    # buffer = VectorReplayBuffer.load_hdf5(filepath, device='cuda')
    buffer = ReplayBuffer.load_hdf5(filepath, device='cuda')

    # print(len(buffer))
def record_step(init_obs, current_action_agent, obs, reward, terminated, truncated):

    buffer.add(Batch(obs=init_obs, act=current_action_agent,
                     rew=reward, terminated=terminated , truncated = truncated, obs_next=obs))


    # Prepare data in a format compatible with Tianshou's Batch
    # step_data = Batch(
    #     obs=[init_obs],
    #     act=[current_action_agent],
    #     obs_next=[obs],
    #     rew=[reward],
    #     terminated=[terminated], truncated=[truncated]
    # )
    # # print(step_data)
    # # Add the data for each agent; adjust this according to your specific scenario
    # buffer.add(step_data, buffer_ids=[0])

def save_buffer_to_hdf5(file_path):
    buffer.save_hdf5(file_path)



# Game loop
running = True
init_obs  = observation

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in agent1_actions:
                current_action_agent1 = agent1_actions[event.key]
            elif event.key in agent2_actions:
                current_action_agent2 = agent2_actions[event.key]
    time.sleep(0.1)
    env.render()



    # Environment interaction logic
    # # Simulate environment step with the current actions of both agents
    # action = {'agent_1':(current_action_agent1),
    #           'agent_2': (current_action_agent2)}
    observation, reward, term, truncated, info  = env.step(current_action_agent1+1)
    observation2, reward2, term2, truncated2, info2 = env.step(current_action_agent2+1)
    if not (reward[0]==0 and reward[1] ==0):
        print('reward:',reward)
    if not (reward2[0]==0 and reward2[1] ==0):
        print('reward2:',reward2)




    # Record the step
    record_step(init_obs, current_action_agent1+1,  observation, reward, term,truncated)
    record_step(observation, current_action_agent2+1, observation2, reward2, term2,truncated2)


    if term2 or truncated2:
        observation = env.reset()  # Reset the environment for the next episode
        data_records = []
        save_buffer_to_hdf5(filepath)  # Save recorded data at the end of each episode

    current_action_agent2 = -1
    current_action_agent1 = -1
    init_obs = observation2


# Cleanup
pygame.quit()
env.close()
sys.exit()
