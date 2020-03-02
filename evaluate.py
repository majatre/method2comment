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
from dpu_utils.utils import run_and_debug, RichPath

from models.model_main import LanguageModel
from data_processing.metrics import calculate_metrics 
from data_processing.method2comment_dataset import JsonLMethod2CommentDataset
from tf2_gnn.data import DataFold, GraphDataset

import pickle
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used


def run(arguments) -> None:
    print("Loading data ...")
    dataset_params = JsonLMethod2CommentDataset.get_default_hyperparameters()
    dataset = JsonLMethod2CommentDataset(dataset_params)
    data_description = dataset.get_batch_tf_data_description()

    model = LanguageModel.restore(arguments["TRAINED_MODEL"], data_description.batch_features_shapes)
    print(f"  Loaded trained model from {arguments['TRAINED_MODEL']}.")

    dataset.load_vocab(model.vocab_source, model.vocab_target)
    data_path = RichPath.create(
        os.path.join(os.path.dirname(__file__), ".", "jsonl_datasets/" + args['TEST_DATA_DIR'])
    )
    dataset.load_data(data_path, folds_to_load=[DataFold.VALIDATION])
    test_data = dataset.get_tensorflow_dataset(DataFold.VALIDATION)

    print(
        f"  Loaded {len(list(test_data))} testing samples."
    )

    test_loss, test_acc, test_true, test_pred = model.run_one_epoch(
            test_data,
            training=False,
        )
      
    test_bleu, test_nist, test_dist, test_rouge2, test_rougel = calculate_metrics(test_true, test_pred)
    print(f"Test:  Loss {test_loss:.4f}, Acc {test_acc:.3f}, BLEU {test_bleu:.3f}")
    print(f"       NIST {test_nist:.3f}, DIST {test_dist:.3f}, ROUGE-2 {test_rouge2:.3f}, ROUGE-L {test_rougel:.3f}")

    

if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
