#!/usr/bin/env python
"""
Usage:
    train.py [options] SAVE_DIR TRAIN_DATA_DIR VALID_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help                        Show this screen.
    --max-num-epochs EPOCHS          The maximum number of epochs to run [default: 500]
    --patience NUM                   Number of epochs to wait for model improvement before stopping [default: 5]
    --max-num-files INT              Number of files to load.
    --hypers-override HYPERS         JSON dictionary overriding hyperparameter values.
    --run-name NAME                  Picks a name for the trained model.
    --debug                          Enable debug routines. [default: False]
"""
import json
import os
import time
from typing import Dict, Any
import datetime

import tensorflow as tf
import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

from models.model_main import LanguageModel
from data_processing.metrics import calculate_metrics 
from data_processing.method2comment_dataset import JsonLMethod2CommentDataset
from tf2_gnn.data import DataFold #, JsonLMethod2CommentDataset 

import pickle


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

def jsonl_dataset(dataset_name: str):
    dataset_params = JsonLMethod2CommentDataset.get_default_hyperparameters()
    dataset = JsonLMethod2CommentDataset(dataset_params)
    data_path = RichPath.create(
        os.path.join(os.path.dirname(__file__), ".", "jsonl_datasets/" + dataset_name)
    )
    dataset.load_data(data_path, folds_to_load=[DataFold.TRAIN, DataFold.VALIDATION])

    return dataset


def train(
    model: LanguageModel,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    batch_size: int,
    max_epochs: int,
    patience: int,
    save_file: str,
    data_description
):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    valid_log_dir = 'logs/' + current_time + '/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    best_valid_loss, best_valid_acc, _, _ = model.run_one_epoch(
        valid_data,
        training=False,
    )
    print(f"Initial valid loss: {best_valid_loss:.3f}.")
    model.save(save_file)
    best_valid_epoch = 0
    train_time_start = time.time()
    for epoch in range(1, max_epochs + 1):
        print(f"== Epoch {epoch}")
        train_loss, train_acc, train_true, train_pred = model.run_one_epoch(
            train_data,
            training=True,
        )
        train_bleu, train_nist, train_dist, train_rouge2, train_rougel = calculate_metrics(train_true, train_pred)
        print(f" Train:  Loss {train_loss:.4f}, Acc {train_acc:.3f}, BLEU {train_bleu:.3f}, NIST {train_nist:.3f},")
        print(f"       NIST {train_nist:.3f}, DIST {train_dist:.3f}, ROUGE-2 {train_rouge2:.3f}, ROUGE-L {train_rougel:.3f}")

        with train_summary_writer.as_default():
          tf.summary.scalar('loss', train_loss, step=epoch)
          tf.summary.scalar('accuracy', train_acc, step=epoch)
          for metric_name, metric_score in zip(["bleu", "nist", "dist", "rouge2", "rougel"], 
            [train_bleu, train_nist, train_dist, train_rouge2, train_rougel]):
            tf.summary.scalar(metric_name, metric_score, step=epoch)

        valid_loss, valid_acc, valid_true, valid_pred = model.run_one_epoch(
            valid_data,
            training=False,
        )
        valid_bleu, valid_nist, valid_dist, valid_rouge2, valid_rougel = calculate_metrics(valid_true, valid_pred)
        print(f" Valid:  Loss {valid_loss:.4f}, Acc {valid_acc:.3f}, BLEU {valid_bleu:.3f}")
        print(f"       NIST {valid_nist:.3f}, DIST {valid_dist:.3f}, ROUGE-2 {valid_rouge2:.3f}, ROUGE-L {valid_rougel:.3f}")

        with valid_summary_writer.as_default():
          tf.summary.scalar('loss', valid_loss, step=epoch)
          tf.summary.scalar('accuracy', valid_acc, step=epoch)
          for metric_name, metric_score in zip(["bleu", "nist", "dist", "rouge2", "rougel"], 
            [valid_bleu, valid_nist, valid_dist, valid_rouge2, valid_rougel]):
            tf.summary.scalar(metric_name, metric_score, step=epoch)

        # Save if good enough.
        if valid_acc >= best_valid_acc:
            print(
                f"  (Best epoch so far, acc increased to {valid_acc:.4f} from {best_valid_acc:.4f})",
            )
            model.save(save_file)
            print(f"  (Saved model to {save_file})")
            best_valid_acc = valid_acc
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            print(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation acc.",
            )
            print(
                f"Training took {total_time:.0f}s. Best validation acc: {best_valid_acc:.4f}",
            )
            break


def run(arguments) -> None:
    hyperparameters = LanguageModel.get_default_hyperparameters()
    hyperparameters["run_id"] = make_run_id(arguments)
    max_epochs = int(arguments.get("--max-num-epochs"))
    patience = int(arguments.get("--patience"))
    max_num_files = arguments.get("--max-num-files")

    # override hyperparams if flag is passed
    hypers_override = arguments.get("--hypers-override")
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))

    save_model_dir = args["SAVE_DIR"]
    os.makedirs(save_model_dir, exist_ok=True)
    save_file = os.path.join(
        save_model_dir, f"{hyperparameters['run_id']}_best_model.bin"
    )

    print("Loading data ...")
    dataset = jsonl_dataset(args['TRAIN_DATA_DIR'])
    # tf_dataset = dataset.get_tensorflow_dataset(DataFold.TRAIN, use_worker_threads=False)
    data_description = dataset.get_batch_tf_data_description()
    print(data_description.batch_features_shapes)

    vocab_source = dataset.vocab_source
    vocab_target = dataset.vocab_target
    print(f"  Built source vocabulary of {len(vocab_source)} entries.")
    print(f"  Built comment vocabulary of {len(vocab_target)} entries.")
    train_data = dataset.get_tensorflow_dataset(DataFold.TRAIN)
    print(f"  Loaded {len(list(train_data))} training samples from {args['TRAIN_DATA_DIR']}.")
    valid_data = dataset.get_tensorflow_dataset(DataFold.VALIDATION)
    print(f"  Loaded {len(list(valid_data))} validation samples from {args['VALID_DATA_DIR']}.")
    model = LanguageModel(hyperparameters, vocab_source, vocab_target)
    model.build(data_description.batch_features_shapes)
    print(
        f"Constructed model, using the following hyperparameters: {json.dumps(hyperparameters)}"
    )

    train(
        model,
        train_data,
        valid_data,
        batch_size=hyperparameters["batch_size"],
        max_epochs=max_epochs,
        patience=patience,
        save_file=save_file,
        data_description=data_description,
    )


def make_run_id(arguments: Dict[str, Any]) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    user_save_name = arguments.get("--run-name")
    if user_save_name is not None:
        user_save_name = (
            user_save_name[: -len(".pkl")]
            if user_save_name.endswith(".pkl")
            else user_save_name
        )
        return "%s" % (user_save_name)
    else:
        return "RNNModel-%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))


if __name__ == "__main__":
    
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
