from keras.layers import Input
from keras.models import load_model, Model
training_model = load_model('mind.h5')

encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output 
encoder_states = [state_h_enc, state_c_enc]
encoder_model =  Model(encoder_inputs, encoder_states)

latent_dim = 256 
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
