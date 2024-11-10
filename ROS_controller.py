# ROS_controller.py
import rospy
from std_msgs.msg import ByteMultiArray, Int32, Bool
import subprocess
import json

class RosSubPub:
    def __init__(self):
        rospy.init_node('state_reader')
        self.tag_table_subscriber = rospy.Subscriber('tag_table', ByteMultiArray, callback=self.refresh_tags_tables)
        self.task_num_publisher = rospy.Publisher('task_num', Int32, queue_size=10)
        self.task_done_subscriber = rospy.Subscriber('task_done', Bool, callback=self.receive_task_done)
        self.state = None
        self.done = False

    def refresh_tags_tables(self, array):
        self.state = [array.data] + [1]

    def receive_task_done(self, data):
        self.done = data.data

def call_compute_action(observation, info=None, state=None):
    # Use subprocess to call ROS_compute_action.py in another Conda environment
    observation_json = json.dumps(observation)  # Serialize observation for passing
    result = subprocess.run(
        ["conda", "run", "-n", "compute_env", "python", "ROS_compute_action.py", observation_json],
        capture_output=True,
        text=True
    )
    # Parse the output
    action = int(result.stdout.strip())
    return action

if __name__ == "__main__":
    ros_sub_pub = RosSubPub()
    running = True
    while running:
        if ros_sub_pub.done:
            observation = ros_sub_pub.state
            action_robot = call_compute_action(observation)
            ros_sub_pub.task_num_publisher.publish(Int32(action_robot - 1))
            human_action = input("Enter human action (press 'q' to quit): ")
            print('Human action:', human_action)
            if human_action == 'q':
                break
