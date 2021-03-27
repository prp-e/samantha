import numpy as np 
import random

human_data = "human.txt"
robot_data = "robot.txt"

with open(human_data) as chats_1:
    human_chats = chats_1.read().split('\n')

with open(robot_data) as chats_2:
    robot_chats = chats_2.read().split('\n')

print(f'Human chats: {len(human_chats)}, Robot chats: {len(robot_chats)}')

pairs = list(zip(human_chats, robot_chats))


