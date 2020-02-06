import tensorflow.compat.v2 as tf
import sys

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, hyperparameters):
    super(Decoder, self).__init__()
    self.hyperparameters = hyperparameters
    self.target_vocab_size = vocab_size

    # Define embedding layer
    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, self.hyperparameters["token_embedding_size"]) 
    # Define RNN on embedded tokens
    self.lstm = tf.keras.layers.LSTM(self.hyperparameters["decoder_rnn_hidden_dim"],
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')
    # Define fully connected layer to project RNN outputs onto the vocabulary to obtain logits.
    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, token_ids: tf.Tensor, hidden):
    # Embed tokens
    embedded = self.embedding_layer(token_ids)
    # Run RNN on embedded tokens
    output, state_h, state_c = self.lstm(embedded, initial_state = hidden)
    # Project RNN outputs onto the vocabulary to obtain logits.
    rnn_output_logits = self.fc(output) 

    return rnn_output_logits, [state_h, state_c]