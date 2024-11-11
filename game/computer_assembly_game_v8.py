import pygame
import time
import  random
import pandas as pd
# make it to running into multiple cycles.


class ComputerAssemblyGame:
    def __init__(self):
        pygame.init()
        self.screen_width = 1500
        self.screen_height = 1000
        self.grid_size = 100  # 设置网格大小为100像素
        self.num_cycle = 5
        self.count_cycle = 0


        self.component_names = ['1.motherboard', '2.CPU', '3.cooler', '4.GPU', '5.memory_card', '6.power_supply', '7.hard_disk', '8.cover']
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.case_x, self.case_y = 0, 200  # 机箱的位置
        self.case_width, self.case_height = 400, 400  # 机箱的大小
        self.messages = []  # 存储要显示的消息及其时间戳


        # self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        # pygame.display.set_caption("电脑装配游戏")
        # self.clock = pygame.time.Clock()
        #
        # self.background_image = pygame.transform.scale(pygame.image.load('game/image/background.png'), (self.screen_width, self.screen_height))
        # self.case_image = pygame.transform.scale(pygame.image.load('game/image/case.png'),(self.case_width,self.case_height))
        # self.last_keys = pygame.key.get_pressed()



        self.init_position = {}
        for component in self.component_names:
            self.init_position[component] = (0, 0)


        # 初始化手的位置和图像
        self.initialize_hands()

        # 初始化零件
        self.initialize_components()
        self.reward_agent1 = 0 # update it every action
        self.reward_agent2 = 0 # update it every action
        self.reward_agent1_part1_step = 0
        self.reward_agent1_part2_completion  = 0
        self.reward_agent1_part3_failAction = 0
        self.reward_agent1_part4_fatigue = 0

        self.fail_action_reward = -10
        self.assemble_success_reward = 0
        self.alpha = 20

        self.completion_reward = 10
        self.total_step = 0
        self.action_hand1 = None
        self.action_hand2 = None
        self.target_x1 = self.target_y1 = self. target_x2 = self.target_y2 = None
        self.hand2_waiting_for_handover = False
        self.terminated = False
        self. steps_hand1 = self.steps_hand2 = 0
        self.running = True
        # 初始化组件状态
        self.states = {}
        for component in self.component_names:
            self.states[component] = {'position': (0, 0), 'state': 1}

        self.log_file_path = 'game/data/state_log.txt'
        self.write_log_header()
        self.action_log_file_path = 'game/data/action_log.txt'
        self.create_action_log_file()
        # 初始化树结构
        self.initialize_tree()
        excel_file = './assets/random_numbers.xlsx'  # Replace with the path to your Excel file
        # excel_file = './assets/feiqiang_muscle_fatigue_results.xlsx'
        self.df = pd.read_excel(excel_file)
        self.accumulate_fatigue = pd.Series(0, index=self.df.iloc[1].index)
        # Requesting executable tasks
        self.human_capability = [1, 1, 1, 1, 1, 1, 1, 1]
        self.robot_capability = [0, 1, 1, 1, 1, 1, 1, 0]
    def next_round(self):
        # looks like similar to reset game except the acc_fatigue.
        self.messages.clear()  # Clear any existing messages
        # Reset states for all components
        for component in self.component_names:
            self.states[component] = {'position': (0, 0), 'state': 1}
        for component in self.component_names:
            self.init_position[component] = (0, 0)


        # Reset hands positions
        self.initialize_hands()

        # Reset components to initial positions and states
        self.initialize_components()

        # Reset the handover waiting state
        self.hand2_waiting_for_handover = False
        self.action_hand1 = None
        self.action_hand2 = None
        self.target_x1 = self.target_y1 = self.target_x2 = self.target_y2 = None
        self.hand2_waiting_for_handover = False
        self.steps_hand1 = self.steps_hand2 = 0

        # Re-initialize the game tree if necessary
        self.initialize_tree()
        # self.render()
    def reset_game(self, seed):
        # Reset the game state to initial values
        self.messages.clear()  # Clear any existing messages
        # Reset states for all components
        for component in self.component_names:
            self.states[component] = {'position': (0, 0), 'state': 1}
        for component in self.component_names:
            self.init_position[component] = (0, 0)


        # Reset hands positions
        self.initialize_hands()

        # Reset components to initial positions and states
        self.initialize_components()

        # Reset the handover waiting state
        self.hand2_waiting_for_handover = False



        # Reset game-specific variables
        self.reward_agent1 = 0 # update it every action
        self.reward_agent2 = 0 # update it every action
        self.accumulate_fatigue = pd.Series(0, index=self.df.iloc[1].index)
        self.action_hand1 = None
        self.action_hand2 = None
        self.target_x1 = self.target_y1 = self.target_x2 = self.target_y2 = None
        self.hand2_waiting_for_handover = False
        self.terminated = False
        self.steps_hand1 = self.steps_hand2 = 0
        self.count_cycle = 0

        # Re-initialize the game tree if necessary
        self.initialize_tree()
        # self.render()

        # Clear logs or reinitialize log files if logging is required
        # self.write_log_header()
        # self.create_action_log_file()
    def move_with_random_action(self, action_hand1, action_hand2, human_lazy_hard):
        executable_tasks = self.get_executable_tasks()
        if human_lazy_hard == 0:
            do_human = random.randint(1,10)
        else:
            do_human = 10
        rand_task = 0
        if len(executable_tasks) ==0:
            rand_task = -1
        else:
            for i in range(len(executable_tasks)):
                if int(executable_tasks[i][0]) == action_hand2:
                    executable_tasks.pop(i)
                    break
            if len(executable_tasks) == 0:
                rand_task = -1
            else:
                rand_task = random.randint(0,len(executable_tasks)-1)
        if do_human <= 4:
            return self.move_hands_simultaneously( action_hand2, -1)
        else:
            if rand_task == -1:
                return self.move_hands_simultaneously( action_hand2, -1)
            else:
                task_name = executable_tasks[rand_task]
                if int(task_name[0]) == action_hand2:
                    return self.move_hands_simultaneously(action_hand2, -1)
                else:
                    return self.move_hands_simultaneously(action_hand2, int(task_name[0]))

    def move_hands_simultaneously(self, action_hand1, action_hand2):
        # exchange the parameter of action hand2 and hand 1 in the def line.
        print('action1', action_hand1, 'action2', action_hand2)
        self.reward_agent1 = 0 # update it every action
        self.reward_agent2 = 0 # update it every action
        self.reward_agent1_part1_step = 0
        self.reward_agent1_part2_completion  = 0
        self.reward_agent1_part3_failAction = 0
        self.reward_agent1_part4_fatigue = 0
        self.reward_agent1_part1_step -=1
        self.reward_agent1 -=1
        self.reward_agent2 -=1
        if self.action_hand1 is None and action_hand1 !=-1 and action_hand1 !=0:
            self.action_hand1 = action_hand1
            if 1 <= action_hand1 <= 8:
                self.target_x1, self.target_y1 = self.components[self.action_hand1-1].x, self.components[self.action_hand1-1].y
            elif action_hand1 == 9:
                self.target_x1, self.target_y1 = self.case_x, self.case_y  # Set target to case position for Hand1
            if self.target_x1 is not None:
                self.hand1_operate()

        elif self.action_hand1 is not None:
            # move one step or pick or drop an item
            self.hand1_operate()

        if self.action_hand2 is None and action_hand2!=-1 and action_hand2!= 0:
            self.action_hand2 = action_hand2
            if 1 <= action_hand2 <= 8:
                component_id2 = action_hand2 - 1
                self.target_x2, self.target_y2 = self.components[component_id2].x, self.components[component_id2].y
            elif action_hand2 ==9:
                self.target_x2, self.target_y2 = self.case_x, self.case_y  # Set target to case position for Hand2
            if self.target_x2 is not None:
                self.hand2_operate()
        elif self.action_hand2 is not None:
            self.hand2_operate()

        self.update()
        reward_agent1_parts =  [self.reward_agent1_part1_step,self.reward_agent1_part2_completion, self.reward_agent1_part3_failAction, self.reward_agent1_part4_fatigue]

        return [self.reward_agent1, self.reward_agent2], self.terminated,reward_agent1_parts
    def hand1_operate (self):
        num_step = 0
        while True:
            num_step +=1
            self.hand_x, self.hand_y, step_made_1 = self.move_towards(
                self.hand_x, self.hand_y, self.target_x1, self.target_y1, 'y')
            self.total_step += 1
            if step_made_1 == 0:
                break

        if not step_made_1:
            if 1 <= self.action_hand1 <= 8:
                self.assemble_part(1, self.action_hand1-1)
            self.target_x1 = None
            self.target_y1 = None
            self.action_hand1 = None
        self.total_step += 1
    def hand2_operate(self):
        while True:
            self.second_hand_x, self.second_hand_y, step_made2 = self.move_towards(
                self.second_hand_x, self.second_hand_y, self.target_x2, self.target_y2, 'y')
            self.total_step += 1
            if step_made2 == 0:
                break

        if not step_made2:
            self.assemble_part(2, self.action_hand2 - 1)

            self.target_x2 = None
            self.target_y2 = None
            self.action_hand2 = None

    def reward_assembly(self, comp):
        # cost = (abs(comp.rect.left - self.case_x) + abs(comp.rect.top-self.case_y))/self.grid_size
        cost = (abs(self.init_position[comp.type][0] - self.case_x) + abs(self.init_position[comp.type][1]-self.case_y))/self.grid_size

        print('cost:',cost)
        return 10 if cost >= 12 else -10

    def assemble_part(self, hand_id, num_comp):
        """
        hand1 assemble the object
        hand2 move to the position of case and wait for handover

        """
        executable_tasks = self.get_executable_tasks()
        is_executable = False
        for x in executable_tasks:
            if int(x[0]) == int(num_comp) +1:
                is_executable = True


        if hand_id == 2:
            if not (self.human_capability[num_comp]==1 and is_executable):
                self.reward_agent2 += self.fail_action_reward
                return
        else:
            if not(self.robot_capability[num_comp] == 1 and is_executable):
                self.reward_agent1 += self.fail_action_reward
                self.reward_agent1_part3_failAction += self.fail_action_reward
                return
        comp = self.components[num_comp]
        executable_tasks = self.get_executable_tasks()
        if (comp.type in executable_tasks) and (self.states[comp.type]['state'] == 1):
            self.states[comp.type]['position'] = (0, 0)
            comp.x, comp.y = (0, 0)
            self.states[comp.type]['state'] = 4  # in hand1
            leaf_node = self.find_leaf_by_name(self.root, comp.type)
            result = leaf_node.execute()
            result = leaf_node.complete()

            if hand_id == 1:
                print('robot assemble')
                self.reward_agent1 += self.assemble_success_reward

            else:
                print('human assemble')
                self.reward_agent2 += self.assemble_success_reward

                # self.reward_agent1 = self.reward_agent1 - self.df.iloc[num_comp].sum() /20  *self.alpha - self.df.iloc[
                #     num_comp].max() *self.alpha
                old_acc_fatigue  = self.accumulate_fatigue
                self.accumulate_fatigue  =  self.accumulate_fatigue + self.df.iloc[num_comp]
                punish_fatigue_reward = self.alpha * (self.accumulate_fatigue.sum()/20 + self.accumulate_fatigue.max() - old_acc_fatigue.sum()/20 - old_acc_fatigue.max())
                self.reward_agent1 = self.reward_agent1 - punish_fatigue_reward
                self.reward_agent1_part4_fatigue -= punish_fatigue_reward

            if comp.type == '8.cover':
                self.reward_agent1 += self.completion_reward
                self.reward_agent2 += self.completion_reward
                self.reward_agent1_part2_completion += self.completion_reward
                self.count_cycle += 1
                if self.count_cycle == self.num_cycle:
                    self.terminated = True
                else:
                    self.next_round()
        else:
            if hand_id ==1:
                self.reward_agent1 += self.fail_action_reward
                self.reward_agent1_part3_failAction += self.fail_action_reward
            else:
                self.reward_agent2 += self.fail_action_reward



    def pick_up_component(self, component_id, hand_id):
        """
        Marks a component as picked up by a specific hand.
        """
        component = self.components[component_id]
        if hand_id == 1:
            if self.human_capability[component_id]==0:
                self.reward_agent1 += self.fail_action_reward
                self.reward_agent1_part3_failAction += self.fail_action_reward

                return
        else:
            if self.robot_capability[component_id]==0:
                self.reward_agent2 += self.fail_action_reward
                return

        if hand_id ==1:
            for comp in self.components: # in case hand1 pick up two items.
                if comp.picked_up_by_hand1:
                    comp.picked_up_by_hand1 = False
                    leaf_node = self.find_leaf_by_name(self.root, comp.type)
                    leaf_node.drop()
                    self.states[comp.type]['state'] = 1
                    self.reward_agent1 += self.fail_action_reward
                    self.reward_agent1_part3_failAction += self.fail_action_reward


        else:
            for comp in self.components: # same to hand1, can't pick up two items.
                if comp.picked_up_by_hand2:
                    comp.picked_up_by_hand2 = False
                    leaf_node = self.find_leaf_by_name(self.root, comp.type)
                    leaf_node.drop()
                    self.states[comp.type]['state'] = 1
                    self.reward_agent2 += self.fail_action_reward


        result =''
        if self.states[component.type]['state'] ==1:
            leaf_node = self.find_leaf_by_name(self.root, component.type)
            result = leaf_node.execute()
            if result == 'success':
                if hand_id == 1:
                    component.picked_up_by_hand1 = True
                else:
                    component.picked_up_by_hand2 = True

                self.states[component.type]['state'] = 2  # in hand1
                self.states[component.type]['position'] = (component.x, component.y)
                # You can add additional logic here, such as updating the game state to reflect the component being held by the hand
                # print(f"Component {component_id + 1} picked up by Hand {hand_id}.")
            else:
                if hand_id ==1:
                    self.reward_agent1 += self.fail_action_reward
                    self.reward_agent1_part3_failAction += self.fail_action_reward

                else:
                    self.reward_agent2 += self.fail_action_reward

        else:
            # print(f"Hand {hand_id} fail to pick up Component {component_id + 1}.")
            if hand_id == 1:
                self.reward_agent1 += self.fail_action_reward
                self.reward_agent1_part3_failAction += self.fail_action_reward
            else:
                self.reward_agent2 += self.fail_action_reward



    def move_towards(self, start_x, start_y, target_x, target_y, last_direction):
        """
        Moves from start to target by one grid size in either x or y direction, not both.
        last_direction indicates the last direction of movement ('x' or 'y'), to alternate movement direction.
        """
        step_made = False
        new_direction = ''

        if last_direction == 'y' or last_direction == '':  # Prioritize x-direction first if last move was y or if starting
            if start_x < target_x:
                distance_step = min(self.grid_size, target_x-start_x)
                start_x += distance_step  # Move right
                step_made = True
                new_direction = 'x'
            elif start_x > target_x:
                distance_step = min(self.grid_size, start_x- target_x)
                start_x -= distance_step  # Move left
                step_made = True
                new_direction = 'x'
        if (start_y < target_y or start_y > target_y) and not step_made:  # Then move in y if x didn't change or was not prioritized
            if start_y < target_y:
                distance_step = min(self.grid_size, target_y-start_y)
                start_y += distance_step  # Move down
                step_made = True
                new_direction = 'y'
            elif start_y > target_y:
                distance_step = min(self.grid_size, start_y- target_y)
                start_y -= self.grid_size  # Move up
                step_made = True
                new_direction = 'y'

        # Return the updated position, whether a step was made, and the direction of the move
        return start_x, start_y, step_made

    def initialize_components(self):
        self.components = []
        used_positions = set()  # Track used grid positions

        image_paths = ['game/image/motherboard.png', 'game/image/cpu.png', 'game/image/cooler.png', 'game/image/gpu.png',
                       'game/image/memory_card.png', 'game/image/power_supply.png', 'game/image/hard_disk.png',
                       'game/image/cover.png']

        image_path = 'game/image/motherboard.png'
        for name, image_path2 in zip(self.component_names, image_paths):
            while True:
                grid_x = random.randint(4, (self.screen_width - self.grid_size) // self.grid_size)
                grid_y = random.randint(0, (self.screen_height - self.grid_size) // self.grid_size)
                x = grid_x * self.grid_size
                y = grid_y * self.grid_size
                if (grid_x, grid_y) not in used_positions:
                    used_positions.add((grid_x, grid_y))
                    break
            self.components.append(Component(x, y, self.grid_size, self.grid_size, name, image_path))
            self.init_position[name] = (x,y)


    def update(self):

        move_distance = self.grid_size  # Move by one grid cell


        # Update hand rects for collision detection
        self.hand1_rect.topleft = (self.hand_x, self.hand_y)
        self.hand2_rect.topleft = (self.second_hand_x, self.second_hand_y)


        for component in self.components:
            if component.picked_up_by_hand1:
                component.x = self.hand_x
                component.y = self.hand_y
            elif component.picked_up_by_hand2:
                component.x = self.second_hand_x
                component.y = self.second_hand_y




        executable_tasks = self.get_executable_tasks()

    def initialize_hands(self):
        self.hand_x, self.hand_y = 400, 300
        self.second_hand_x, self.second_hand_y = 200, 300

        self.hand1_image = pygame.transform.scale(pygame.image.load("game/image/hand.png"), (30, 30))
        self.hand2_image = pygame.transform.scale(pygame.image.load("game/image/robot.png"), (30, 30))

        self.hand1_rect = self.hand1_image.get_rect(topleft=(self.hand_x, self.hand_y))
        self.hand2_rect = self.hand2_image.get_rect(topleft=(self.second_hand_x, self.second_hand_y))
    def add_message(self, message):
        """ 添加消息及其时间戳 """
        current_time = time.time()
        self.messages.append((message, current_time))

    def update_messages(self):
        """ 更新消息列表，移除过期的消息 """
        current_time = time.time()
        self.messages = [(msg, timestamp) for msg, timestamp in self.messages if current_time - timestamp < 1]





    def run(self):
        while self.running:
            self.update()
            self.render()
            self.clock.tick(60)

        pygame.quit()

    def distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def draw_text(self, text, position, color=(255, 255, 255), font_size=64):
        font = pygame.font.SysFont(None, font_size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def render(self):
        self.screen.fill(self.BLACK)
        self.screen.blit(self.background_image, (0, 50))

        self.screen.blit(self.case_image,(self.case_x, self.case_y))

        # 绘制机箱
        # pygame.draw.rect(self.screen, self.case_color, (self.case_x, self.case_y, self.case_width, self.case_height))

        # 绘制零件
        for component in self.components:
            component.draw(self.screen)

        # 绘制手的图片
        self.screen.blit(self.hand1_image, self.hand1_rect)
        self.screen.blit(self.hand2_image, self.hand2_rect)

        executable_tasks = self.get_executable_tasks()
        if executable_tasks:
            # 显示可执行任务
            self.draw_text('Task promopt: ',(10, 10))
            for i, task in enumerate(executable_tasks):

                self.draw_text(task, (10, 10 + (i+1) * 30))
        else:
            # 如果没有可执行任务，显示完成提示
            self.draw_text("None", (10, 10))

        # 更新消息
        self.update_messages()

        # 显示消息
        for message, timestamp in self.messages:
            self.draw_text(message, (10, 500), (255, 0, 0))  # 显示在屏幕下方

        pygame.display.flip()

    def find_leaf_by_name(self, node, name):
        if node.name == name and isinstance(node, LeafNode):
            return node
        for child in node.children:
            found = self.find_leaf_by_name(child, name)
            if found:
                return found
        return None

    def write_log_header(self):
        header = "Time, " + ", ".join([f"{name}_state, {name}_position" for name in self.states.keys()])
        with open(self.log_file_path, 'w') as file:
            file.write(header + '\n')
    def create_action_log_file(self):
        with open(self.action_log_file_path, 'w') as file:
            file.write('Time, Actor, Action\n')
    def log_action(self, actor, action_code):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"{current_time}, {actor}, {action_code}\n"
        with open(self.action_log_file_path, 'a') as file:
            file.write(log_message)





    def write_to_file(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = current_time + ", " + ", ".join([f"{state['state']}, {state['position']}" for state in self.states.values()])
        with open(self.log_file_path, 'a') as file:
            file.write(log_message + '\n')




    def get_executable_tasks(self):
        # 获取可执行任务的逻辑
        incomplete_leaves = self.root.get_all_incomplete_leaf_nodes()
        following_tasks = set()

        for leaf in incomplete_leaves:
            following_tasks.update(leaf.query_following_leaf_nodes())

        executable_tasks = [leaf for leaf in incomplete_leaves if leaf not in following_tasks]
        executable_tasks = [leaf for leaf in executable_tasks if leaf.state != 'doing']

        return [leaf.name for leaf in executable_tasks]
    def initialize_tree(self):
        # 示例树  'motherboard', 'CPU', 'cooler', 'GPU', 'memory_card', 'power_supply', 'hard_disk', 'cover']


        # 创建电脑装配的树结构
        self.root = AndNode('desktop case')

        # 主板安装
        motherboard = LeafNode('1.motherboard')
        self.root.add_child(motherboard)

        motherboard_installation = OrNode('Partsofmainboard')
        self.root.add_child(motherboard_installation)
        cpu_fan  = AndNode('cpu|fan')
        motherboard_installation.add_child(cpu_fan)

        # CPU 安装必须在主板安装之后
        cpu_installation = LeafNode('2.CPU')

        fan_installation = LeafNode('3.cooler')
        cpu_fan.add_child(cpu_installation)
        cpu_fan.add_child(fan_installation)

        # 内存安装在主板安装之后
        ram_installation = LeafNode('5.memory_card')
        motherboard_installation.add_child(ram_installation)





        # 显卡安装在主板安装之后
        gpu_installation = LeafNode('4.GPU')
        motherboard_installation.add_child(gpu_installation)
        others_installation = OrNode('others')
        self.root.add_child(others_installation)

        # 硬盘安装在主板安装之后
        storage_installation = LeafNode('7.hard_disk')
        others_installation.add_child(storage_installation)


        # 电源安装应该在所有主板组件都已安装后进行
        # 但为了表达这一点，我们将它放置在最后一个位置
        psu_installation = LeafNode('6.power_supply')
        others_installation.add_child(psu_installation)

        lid_installation =LeafNode('8.cover')
        self.root.add_child((lid_installation))



# 定义零件类
class Component:
    def __init__(self, x, y, width, height, type,image_path):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type
        self.original_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.original_image, (width, height))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.picked_up_by_hand1 = False
        self.picked_up_by_hand2 = False

    def center(self):
        return center_position(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
        screen.blit(self.image, self.rect)


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


    # 新增一个方法来找到任意深度的祖先节点
    def get_ancestor(self, i):
        if i == 0:
            return self
        if self.parent:
            return self.parent.get_ancestor(i - 1)
        return None
    # 获取当前节点的所有后续兄弟节点
    def get_following_siblings(self):
        if not self.parent:
            return []
        siblings = self.parent.children
        index = siblings.index(self)
        return siblings[index+1:]
    # 获取当前节点的所有叶节点子孙
    def get_all_leaf_nodes(self):
        if isinstance(self, LeafNode):
            return [self]
        leaf_nodes = []
        for child in self.children:
            leaf_nodes.extend(child.get_all_leaf_nodes())
        return leaf_nodes

    def get_all_incomplete_leaf_nodes(self):
        incomplete_leaf_nodes = []
        if isinstance(self, LeafNode) and self.state != 'done':
            incomplete_leaf_nodes.append(self)
        else:
            for child in self.children:
                incomplete_leaf_nodes.extend(child.get_all_incomplete_leaf_nodes())
        return incomplete_leaf_nodes
    def get_all_executing_leaf_nodes(self):
        incomplete_leaf_nodes = []
        if isinstance(self, LeafNode) and self.state == 'doing':
            incomplete_leaf_nodes.append(self)
        else:
            for child in self.children:
                incomplete_leaf_nodes.extend(child.get_all_executing_leaf_nodes())
        return incomplete_leaf_nodes

class AndNode(Node):
    def __init__(self, name):
        super().__init__(name)

class OrNode(Node):
    def __init__(self, name):
        super().__init__(name)

class LeafNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.state = 'to_be_done'

    def execute(self):
        try:
            if self.can_execute():
                self.state = 'doing'
                # print(f"{self.name} doing")
                return "success"
            else:
                Exception(f"无法执行{self.name}，因为并未满足先决条件。")
        except Exception as e:
            print(e)
            return "failure"

    def drop(self):
        try:
            if self.state == 'doing':
                self.state = 'to_be_done'
                # print(f"{self.name} doing")
                return "success"
            else:
                Exception(f"无法drop{self.name}，因为并未满足先决条件。")
        except Exception as e:
            print(e)
            return "failure"

    def complete(self):
        try:
            if self.state != 'doing':
                raise Exception(f"无法完成{self.name}，因为它尚未开始执行。")
            self.state = 'done'
            print(f"{self.name} 已完成")
            return "success"
        except Exception as e:
            print(e)
            return "failure"

    def can_execute(self):
        # 检查所有祖先节点，以确保满足先决条件
        ancestor = self.parent
        ancestor_child= self
        while ancestor:
            # if ancestor.name == 'root':
            #     return True

            if isinstance(ancestor, AndNode):
                prev_sibling = self.get_previous_sibling_at_level(ancestor,ancestor_child)
                if prev_sibling and not all_leaf_nodes_completed(prev_sibling):
                    return False
            ancestor_child = ancestor
            ancestor = ancestor.parent
        return True
    # 得到当前节点的层级
    def get_level(self):
        level = 0
        current = self
        while current.parent:
            level += 1
            current = current.parent
        return level
    # 新增查询特定深度的祖先节点的后续兄弟节点中的叶节点的功能
    def query_following_leaf_nodes(self):
        following_leaf_nodes = []
        ancestor = self.parent
        ancestor_child  = self
        while ancestor:
            # 如果祖先是AndNode，获取下一代的所有后续兄弟节点
            if isinstance(ancestor, AndNode):
                following_siblings = ancestor_child.get_following_siblings()
                for sibling in following_siblings:
                    # 添加sibling的叶节点子孙到结果列表中
                    following_leaf_nodes.extend(sibling.get_all_leaf_nodes())
            ancestor_child = ancestor
            ancestor = ancestor.parent
        return following_leaf_nodes

    # 找到在指定层级的前一个兄弟节点
    def get_previous_sibling_at_level(self, ancestor,ancestor_child):
        # parent = self.get_ancestor(self.get_level() - ancestor.get_level())
        parent = ancestor
        index = parent.children.index(ancestor_child)
        if index > 0:
            return parent.children[index - 1]
        return None


def all_leaf_nodes_completed(node):
    if isinstance(node, LeafNode):
        return node.state == 'done'
    return all(all_leaf_nodes_completed(child) for child in node.children)
def check_completion(node):
    if isinstance(node, LeafNode) and node.state != 'done':
        return False
    for child in node.children:
        if not check_completion(child):
            return False
    return True

# 定义手和主板的中心点位置
def center_position(x, y, width, height):
    return x + width // 2, y + height // 2




if __name__ == "__main__":
    game = ComputerAssemblyGame()
    game.run()
