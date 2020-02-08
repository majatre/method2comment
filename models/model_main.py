import os
import pickle
from typing import Dict, Any, NamedTuple, Iterable, List

import numpy as np
import tensorflow.compat.v2 as tf
from dpu_utils.mlutils.vocabulary import Vocabulary

from models import GRU_encoder
from models import GRU_decoder
from models import LSTM_encoder
from models import LSTM_decoder

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel("ERROR")


class LanguageModelLoss(NamedTuple):
    token_ce_loss: tf.Tensor
    num_predictions: tf.Tensor
    num_correct_token_predictions: tf.Tensor


class LanguageModel(tf.keras.Model):
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
            "encoder_rnn_hidden_dim": 64,
            "decoder_rnn_hidden_dim": 64,
            "rnn_cell": "LSTM"  # One of "GRU", "LSTM"
        }

    def __init__(self, hyperparameters: Dict[str, Any], vocab_source: Vocabulary, vocab_target: Vocabulary) -> None:
        self.hyperparameters = hyperparameters
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target

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

        if self.hyperparameters["rnn_cell"] == "GRU":
            self.encoder = GRU_encoder.Encoder(len(self.vocab_source), self.hyperparameters)
            self.decoder = GRU_decoder.Decoder(len(self.vocab_target), self.hyperparameters)
        else: 
            self.encoder = LSTM_encoder.Encoder(len(self.vocab_source), self.hyperparameters)
            self.decoder = LSTM_decoder.Decoder(len(self.vocab_target), self.hyperparameters)

    @property
    def run_id(self):
        return self.hyperparameters["run_id"]

    def save(self, path: str) -> None:
        # We store things in two steps: One .pkl file for metadata (hypers, vocab, etc.)
        # and then the default TF weight saving.
        data_to_store = {
            "model_class": self.__class__.__name__,
            "vocab_source": self.vocab_source,
            "vocab_target": self.vocab_target,
            "hyperparameters": self.hyperparameters,
        }
        with open(path, "wb") as out_file:
            pickle.dump(data_to_store, out_file, pickle.HIGHEST_PROTOCOL)
        self.save_weights(path, save_format="tf")

    @classmethod
    def restore(cls, saved_model_path: str) -> "LanguageModelTF2":
        with open(saved_model_path, "rb") as fh:
            saved_data = pickle.load(fh)

        model = cls(saved_data["hyperparameters"], saved_data["vocab_source"], saved_data["vocab_target"])
        model.build(tf.TensorShape([None, None, None]))
        model.load_weights(saved_model_path)
        return model

    def build(self, input_shape):
        # A small hack necessary so that train.py is completely framework-agnostic:
        input_shape = tf.TensorShape(input_shape)

        super().build(input_shape)

    def call(self, data, training):
        inputs = data[0] 
        target_token_seq = data[1] 
        return self.compute_logits(inputs, target_token_seq, training)

    def compute_logits(self, token_ids: tf.Tensor, target_token_seq: tf.Tensor, training: bool) -> tf.Tensor:
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
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(token_ids, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.vocab_target.get_id_or_unk("%START%")] * self.hyperparameters["batch_size"], 1)

        if training:
          # Use teacher forcing
          predictions, dec_hidden = self.decoder(target_token_seq[:,:-1], dec_hidden)
          return predictions
        else:
          # The predicted ID is fed back into the model
          for t in range(1, self.hyperparameters["max_seq_length"]):
              predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
              predicted_ids = tf.argmax(predictions[:,0,:], 1)
              dec_input = tf.expand_dims(predicted_ids, 1)
              new_logits = tf.expand_dims(predictions[:,0,:], 1)
              if t==1:
                  results = new_logits
              else:
                  results = tf.concat([results, new_logits], 1)

          return results

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
        # token_ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(target_token_seq[:,1:], rnn_output_logits)
        # token_ce_loss = tf.reduce_sum(token_ce_loss)

        # 6# Compute number of (correct) predictions

        # Mask that contains True for the positions of valid tokens
        # and False for the positions of PAD 
        mask = tf.math.not_equal(target_token_seq[:,1:], self.vocab_target.get_id_or_unk(self.vocab_target.get_pad()))

        num_tokens = tf.math.count_nonzero(mask)
        prediction = tf.argmax(rnn_output_logits, 2) 
        compared = tf.cast(tf.math.equal(target_token_seq[:,1:], prediction), tf.int32) * tf.cast(mask, tf.int32)
        num_correct_tokens = tf.math.count_nonzero(compared)
       
        # 7# Mask out CE loss for padding tokens
        token_ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(rnn_output_logits, mask),
            labels=tf.boolean_mask(target_token_seq[:,1:], mask))
        token_ce_loss = tf.reduce_sum(token_ce_loss)
        
        return LanguageModelLoss(token_ce_loss, num_tokens, num_correct_tokens)

    def get_text_from_tensor(self, output: tf.Tensor):
        texts = []
        for token_seq in output:
            texts.append([self.vocab_target.get_name_for_id(t) for t in token_seq])
        return texts

    def predict_single_comment(self, token_seq: List[int]):
        self.hyperparameters["batch_size"] = 1
        output_logits = self.compute_logits(
            np.array([token_seq], dtype=np.int32), training=False
        )
        next_tok_logits = output_logits[0, :, :]
        next_tok_ids = tf.argmax(next_tok_logits, 1).numpy()
        return next_tok_ids
    
    def predict_next_token(self, token_seq: List[int]):
        output_logits = self.compute_logits(
            np.array([token_seq], dtype=np.int32), training=False
        )
        prediction = tf.argmax(output_logits, 2) 
        return prediction.numpy()

    def run_one_epoch(
        self, minibatches: Iterable[np.ndarray], training: bool = False,
    ):
        total_loss, num_samples, num_tokens, num_correct_tokens = 0.0, 0, 0, 0
        ground_truth = []
        predictions = []
        for step, minibatch_data in enumerate(minibatches):
            self.hyperparameters["batch_size"] = len(minibatch_data)
            sources = np.array([x[0] for x in minibatch_data])
            targets = np.array([x[1] for x in minibatch_data])
            with tf.GradientTape() as tape:
                model_outputs = self.compute_logits(sources, targets, training=training)
                result = self.compute_loss_and_acc(model_outputs, targets)

            total_loss += result.token_ce_loss
            num_samples += sources.shape[0]
            num_tokens += result.num_predictions
            num_correct_tokens += result.num_correct_token_predictions

            target_texts = self.get_text_from_tensor(targets)
            predicted_texts = self.get_text_from_tensor(tf.argmax(model_outputs,2))

            ref = [([x[1:x.index("%END%")] if "%END%" in x else x[1:]]) for x in target_texts]
            hyp = [(x[:x.index("%END%")] if "%END%" in x else x) for x in predicted_texts]
            smoothing = SmoothingFunction().method4
            bleu_score = corpus_bleu(ref, predicted_texts, smoothing_function=smoothing)
            # for r, h in zip(ref, hyp):
            #     print('Target', ' '.join(r[0]))
            #     print('Prediction', ' '.join(h))
            #     print('\n')
            # print(bleu_score)

            ground_truth += ref
            predictions += hyp

            if training:
                gradients = tape.gradient(
                    result.token_ce_loss, self.trainable_variables
                )
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            print(
                "   Batch %4i: Epoch avg. loss: %.5f || Batch loss: %.5f | acc: %.5f | bleu: %.5f" 
                % (
                    step,
                    total_loss / num_samples,
                    result.token_ce_loss,
                    float(result.num_correct_token_predictions)
                    / (float(result.num_predictions) + float(1e-7)),
                    bleu_score
                ),
                end="\n",
            )
        print("\r\x1b[K", end="")

        smoothing = SmoothingFunction().method4
        bleu_score = corpus_bleu(ground_truth, predictions, smoothing_function=smoothing)

        return (
            total_loss / num_samples,
            float(num_correct_tokens) / (float(num_tokens) + 1e-7),
            bleu_score
        )
