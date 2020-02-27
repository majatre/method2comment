import tensorflow.compat.v2 as tf

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, hyperparameters):
    super(Encoder, self).__init__()
    self.hyperparameters = hyperparameters

    self.embedding = tf.keras.layers.Embedding(vocab_size, self.hyperparameters["token_embedding_size"])
    self.lstm = tf.keras.layers.LSTM(self.hyperparameters["encoder_rnn_hidden_dim"],
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, token_ids, hidden):
    latent = self.embedding(token_ids)
    output, state_h, state_c = self.lstm(latent, initial_state = hidden)
    return output, [state_h, state_c]

  def initialize_hidden_state(self, batch_size):
    return [tf.zeros((batch_size, self.hyperparameters["encoder_rnn_hidden_dim"])),
            tf.zeros((batch_size, self.hyperparameters["encoder_rnn_hidden_dim"]))]