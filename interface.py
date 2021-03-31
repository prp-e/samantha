from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model

from brain import decoder_dense, decoder_lstm, state_cell, state_hidden, decoder_inputs

training_model = load_model('mind.h5')

encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output 
encoder_states = [state_h_enc, state_c_enc]
encoder_model =  Model(encoder_inputs, encoder_states)

latent_dim = 256 
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

decoder_outputs, state_hidden, state_cell = 
decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)