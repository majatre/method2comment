import tensorflow.compat.v2 as tf

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, hyperparameters):
    super(Encoder, self).__init__()
    self.hyperparameters = hyperparameters

    self.embedding = tf.keras.layers.Embedding(vocab_size, self.hyperparameters["token_embedding_size"])
    self.gru = tf.keras.layers.GRU(self.hyperparameters["encoder_rnn_hidden_dim"],
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, token_ids, hidden):
    latent = self.embedding(token_ids)
    output, state = self.gru(latent, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.hyperparameters["batch_size"], self.hyperparameters["encoder_rnn_hidden_dim"]))