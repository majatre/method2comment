import os
import pickle
from typing import Dict, Any, NamedTuple, Iterable, List

import numpy as np
import tensorflow.compat.v2 as tf
from dpu_utils.mlutils.vocabulary import Vocabulary


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel("ERROR")


class LanguageModelLoss(NamedTuple):
    token_ce_loss: tf.Tensor
    num_predictions: tf.Tensor
    num_correct_token_predictions: tf.Tensor


class LanguageModelTF2(tf.keras.Model):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "optimizer": "Adam",  # One of "SGD", "RMSProp", "Adam"
            "learning_rate": 0.01,
            "learning_rate_decay": 0.98,
            "momentum": 0.85,
            "gradient_clip_value": 1,
            "max_epochs": 500,
            "patience": 5,
            "max_vocab_size": 10000,
            "max_seq_length": 50,
            "batch_size": 200,
            "token_embedding_size": 64,
            "rnn_hidden_dim": 64,
        }

    def __init__(self, hyperparameters: Dict[str, Any], vocab: Vocabulary,) -> None:
        self.hyperparameters = hyperparameters
        self.vocab = vocab

        # Also prepare optimizer:
        optimizer_name = self.hyperparameters["optimizer"].lower()
        if optimizer_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.hyperparameters["learning_rate"],
                momentum=self.hyperparameters["momentum"],
                clipvalue=self.hyperparameters["gradient_clip_value"],
            )
        elif optimizer_name == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSProp(
                learning_rate=self.hyperparameters["learning_rate"],
                decay=self.params["learning_rate_decay"],
                momentum=self.params["momentum"],
                clipvalue=self.hyperparameters["gradient_clip_value"],
            )
        elif optimizer_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.hyperparameters["learning_rate"],
                clipvalue=self.hyperparameters["gradient_clip_value"],
            )
        else:
            raise Exception('Unknown optimizer "%s".' % (self.params["optimizer"]))

        super().__init__()

        # 5# 1) Define embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(len(self.vocab), self.hyperparameters["token_embedding_size"])
        # 5# 2) Define RNN on embedded tokens
        self.gru = tf.keras.layers.GRU(self.hyperparameters["rnn_hidden_dim"],
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        # 5# 3) Define layer to project RNN outputs onto the vocabulary to obtain logits.
        self.output_layer = tf.keras.layers.Dense(len(self.vocab))

    @property
    def run_id(self):
        return self.hyperparameters["run_id"]

    def save(self, path: str) -> None:
        # We store things in two steps: One .pkl file for metadata (hypers, vocab, etc.)
        # and then the default TF weight saving.
        data_to_store = {
            "model_class": self.__class__.__name__,
            "vocab": self.vocab,
            "hyperparameters": self.hyperparameters,
        }
        with open(path, "wb") as out_file:
            pickle.dump(data_to_store, out_file, pickle.HIGHEST_PROTOCOL)
        self.save_weights(path, save_format="tf")

    @classmethod
    def restore(cls, saved_model_path: str) -> "LanguageModelTF2":
        with open(saved_model_path, "rb") as fh:
            saved_data = pickle.load(fh)

        model = cls(saved_data["hyperparameters"], saved_data["vocab"])
        model.build(tf.TensorShape([None, None]))
        model.load_weights(saved_model_path)
        return model

    def build(self, input_shape):
        # A small hack necessary so that train.py is completely framework-agnostic:
        input_shape = tf.TensorShape(input_shape)

        super().build(input_shape)

    def call(self, inputs, training):
        return self.compute_logits(inputs, training)

    def compute_logits(self, token_ids: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Implements a language model, where each output is conditional on the current
        input and inputs processed so far.

        Args:
            token_ids: int32 tensor of shape [B, T], storing integer IDs of tokens.
            training: Flag indicating if we are currently training (used to toggle dropout)

        Returns:
            tf.float32 tensor of shape [B, T, V], storing the distribution over output symbols
            for each timestep for each batch element.
        """
        # 5# 1) Embed tokens
        embedded = self.embedding_layer(token_ids)
        # 5# 2) Run RNN on embedded tokens
        output, state = self.gru(embedded)
        # 5# 3) Project RNN outputs onto the vocabulary to obtain logits.
        rnn_output_logits = self.output_layer(output) 

        return rnn_output_logits

    def compute_loss_and_acc(
        self, rnn_output_logits: tf.Tensor, target_token_seq: tf.Tensor
    ) -> LanguageModelLoss:
        """
        Args:
            rnn_output_logits: tf.float32 Tensor of shape [B, T, V], representing
                logits as computed by the language model.
            target_token_seq: tf.int32 Tensor of shape [B, T], representing
                the target token sequence.

        Returns:
            LanguageModelLoss tuple, containing both the average per-token loss
            as well as the number of (non-padding) token predictions and how many
            of those were correct.
        
        Note:
            We assume that the two inputs are shifted by one from each other, i.e.,
            that rnn_output_logits[i, t, :] are the logits for sample i after consuming
            input t; hence its target output is assumed to be target_token_seq[i, t+1].
        """
        # 5# 4) Compute CE loss for all but the last timestep:
        # Commented out because of the step 7
        # token_ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(target_token_seq[:,1:], rnn_output_logits[:,:-1,:])
        # token_ce_loss = tf.reduce_sum(token_ce_loss)

        # 6# Compute number of (correct) predictions

        # Mask that contains True for the positions of valid tokens
        # and False for the positions of PAD 
        mask = tf.math.not_equal(target_token_seq[:,1:], self.vocab.get_id_or_unk(self.vocab.get_pad()))

        num_tokens = tf.math.count_nonzero(mask)
        prediction = tf.argmax(rnn_output_logits[:,:-1,:], 2) 
        compared = tf.cast(tf.math.equal(target_token_seq[:,1:], prediction), tf.int32) * tf.cast(mask, tf.int32)
        num_correct_tokens = tf.math.count_nonzero(compared)
       
        # 7# Mask out CE loss for padding tokens
        token_ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(rnn_output_logits[:,:-1,:], mask),
            labels=tf.boolean_mask(target_token_seq[:,1:], mask))
        token_ce_loss = tf.reduce_sum(token_ce_loss)
        
        return LanguageModelLoss(token_ce_loss, num_tokens, num_correct_tokens)

    def predict_next_token(self, token_seq: List[int]):
        output_logits = self.compute_logits(
            np.array([token_seq], dtype=np.int32), training=False
        )
        next_tok_logits = output_logits[0, -1, :]
        next_tok_probs = tf.nn.softmax(next_tok_logits)
        return next_tok_probs.numpy()

    def run_one_epoch(
        self, minibatches: Iterable[np.ndarray], training: bool = False,
    ):
        total_loss, num_samples, num_tokens, num_correct_tokens = 0.0, 0, 0, 0
        for step, minibatch_data in enumerate(minibatches):
            with tf.GradientTape() as tape:
                model_outputs = self.compute_logits(minibatch_data, training=training)
                result = self.compute_loss_and_acc(model_outputs, minibatch_data)

            total_loss += result.token_ce_loss
            num_samples += minibatch_data.shape[0]
            num_tokens += result.num_predictions
            num_correct_tokens += result.num_correct_token_predictions

            if training:
                gradients = tape.gradient(
                    result.token_ce_loss, self.trainable_variables
                )
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            print(
                "   Batch %4i: Epoch avg. loss: %.5f || Batch loss: %.5f | acc: %.5f"
                % (
                    step,
                    total_loss / num_samples,
                    result.token_ce_loss,
                    float(result.num_correct_token_predictions)
                    / (float(result.num_predictions) + float(1e-7)),
                ),
                end="\r",
            )
        print("\r\x1b[K", end="")
        return (
            total_loss / num_samples,
            float(num_correct_tokens) / (float(num_tokens) + 1e-7),
        )
