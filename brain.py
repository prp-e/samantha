import numpy as np 
import random
import re 
from tensorflow import keras 
from keras.layers import Input, LSTM, Dense 
from keras.models import Model

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

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

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

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32'
)

decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)

decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)


for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1. 
    for timestep, token in enumerate(target_doc.split()):
        decoder_target_data[line, timestep, target_features_dict[token]] = 1. 
        if timestep > 0: 
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.


dimensionality = 256 
batch_size = 10 
epochs = 1500

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs) 
encoder_states = [state_hidden, state_cell]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
deocder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = deocder_dense(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode = 'temporal')
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split=0.2)
training_model.save('mind.h5')