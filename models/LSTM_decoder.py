import tensorflow.compat.v2 as tf
import sys


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


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
    self.attention = BahdanauAttention(self.hyperparameters["decoder_rnn_hidden_dim"])

  def call(self, token_ids: tf.Tensor, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden[0], enc_output)

    # Embed tokens
    embedded = self.embedding_layer(token_ids)
    # Concat contex from attention
    x = tf.concat([tf.expand_dims(context_vector, 1), embedded], axis=1)
    # Run RNN on embedded tokens
    output, state_h, state_c = self.lstm(x, initial_state = hidden)
    # Project RNN outputs onto the vocabulary to obtain logits.
    # output = tf.reshape(output, (-1, output.shape[2]))
    rnn_output_logits = self.fc(output) 

    return rnn_output_logits, [state_h, state_c]