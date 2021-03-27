import numpy as np 
import random
import re 

human_data = "human.txt"
robot_data = "robot.txt"

with open(human_data) as chats_1:
    human_chats = chats_1.read().split('\n')

with open(robot_data) as chats_2:
    robot_chats = chats_2.read().split('\n')

print(f'Human chats: {len(human_chats)}, Robot chats: {len(robot_chats)}')

human_chats = [re.sub(r"\[w+\]", "سلام", chat) for chat in human_chats]
robot_chats = [re.sub(r"\[w+\]", "سلام", chat) for chat in robot_chats]

pairs = list(zip(human_chats, robot_chats))

input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for pair in pairs: 
    input_doc, target_doc = pair[0], pair[1]
    input_docs.append(input_doc)
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        input_tokens.add(token) 
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)]
)

target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)]
)

reverse_input_features_dict = dict(
    [(i, token) for i, token in input_features_dict.items()]
)

reverse_target_features_dict = dict(
    [(i, token) for i, token in target_features_dict.items()]
)