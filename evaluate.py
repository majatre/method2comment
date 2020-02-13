#!/usr/bin/env python
"""
Usage:
    evaluate.py [options] TRAINED_MODEL TEST_DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from data_processing.dataset import prepare_data, get_minibatch_iterator
from models.model_main import LanguageModel

import pickle

def run(arguments) -> None:
    print("Loading data ...")
    model = LanguageModel.restore(arguments["TRAINED_MODEL"])
    print(f"  Loaded trained model from {arguments['TRAINED_MODEL']}.")

    test_data = pickle.load(open('./data/' + args["TEST_DATA_DIR"] + '.pkl', 'rb'))
    test_data = prepare_data(
        model.vocab_source, model.vocab_target,
        data=test_data,
        max_source_len=model.hyperparameters["max_seq_length"],
        max_target_len=model.hyperparameters["max_seq_length"],
        max_num_files=arguments.get("--max-num-files"),
    )
    print(
        f"  Loaded {test_data.shape[0]} test samples from {arguments['TEST_DATA_DIR']}."
    )

    test_loss, test_acc, bleu = model.run_one_epoch(
        get_minibatch_iterator(
            test_data,
            model.hyperparameters["batch_size"],
            is_training=False,
            drop_remainder=False,
        ),
        training=False,
    )


    print(f"Test:  Loss {test_loss:.4f}, Acc {test_acc:.3f}, BLEU {bleu:.3f}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
