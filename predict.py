#!/usr/bin/env python
"""
Usage:
    predict.py [options] TRAINED_MODEL TOKENS...

Uses trained model to predict comment given a sequence of tokens.

Options:
    -h --help        Show this screen.
    --num-steps NUM  Number of steps to continue token sequence for. [default: 5]
    --debug          Enable debug routines. [default: False]
"""
from typing import List

from docopt import docopt
from dpu_utils.utils import run_and_debug

from data_processing.dataset import tensorise_token_sequence, END_SYMBOL
from models.model_main import LanguageModel


def run(arguments) -> None:
    model = LanguageModel.restore(arguments["TRAINED_MODEL"])

    token_seq = arguments['TOKENS']
    tensorised_seq = tensorise_token_sequence(model.vocab_source, len(token_seq) + 1, token_seq)
    predictions = model.predict_single_comment(tensorised_seq)

    comment = ""
    for token_id in predictions:
        token = model.vocab_target.get_name_for_id(token_id)
        comment += token + " "
    print(comment)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
