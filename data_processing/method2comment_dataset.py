"""General dataset class for datasets with a numeric property stored as JSONLines files."""
from typing import Any, Dict, Iterator, List, Optional, Tuple, Set, Iterable

import logging
import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from tf2_gnn.data.graph_dataset import DataFold, GraphDataset, GraphSampleType, GraphSample, GraphBatchTFDataDescription
from tf2_gnn.data.jsonl_graph_dataset import JsonLGraphDataset

import collections
from dpu_utils.mlutils.vocabulary import Vocabulary

START_SYMBOL = "%START%"
END_SYMBOL = "%END%"

logger = logging.getLogger(__name__)


class GraphWithTargetComment(GraphSample):
    """Data structure holding a single graph with a single numeric property."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: List[np.ndarray],
        target_value: List[np.ndarray],
        source_len: int,
        source_seq: List[np.ndarray],
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._target_value = target_value
        self._source_len = source_len
        self._source_seq = source_seq

    @property
    def target_value(self) -> List[np.ndarray]:
        """Target value of the regression task."""
        return self._target_value

    @property
    def source_len(self):
        """Number of token nodes in the source."""
        return self._source_len

    @property
    def source_seq(self):
        """Token nodes in the source."""
        return self._source_seq


    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"Target_value:   {self._target_value}\n"
            f"Source_len:     {self._source_len}\n"
            f"Source_seq:     {self._source_len}"
        )


class JsonLMethod2CommentDataset(JsonLGraphDataset[GraphWithTargetComment]):
    """
    General class representing pre-split datasets in JSONLines format.
    Concretely, this class expects the following:
    * In the data directory, files "train.jsonl.gz", "valid.jsonl.gz" and
      "test.jsonl.gz" are used to store the train/valid/test datasets.
    * Each of the files is gzipped text file in which each line is a valid
      JSON dictionary with a "graph" key, which in turn points to a
      dictionary with keys
       - "node_features" (list of numerical initial node labels)
       - "adjacency_lists" (list of list of directed edge pairs)
      Addtionally, the dictionary has to contain a "Target" key with a
      floating point value.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers = {
            "num_fwd_edge_types": 3,
            "max_vocab_size": 10000,
            "max_seq_length": 50,
        }
        super_hypers.update(this_hypers)
        return super_hypers

    def __init__(
        self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(params, metadata=metadata)
        self.max_vocab_size = params["max_vocab_size"]
        self.max_seq_length = params["max_seq_length"]
        
    @property
    def node_feature_shape(self) -> Tuple:
        return (1,)

    def load_vocab(self,  vocab_source = None, vocab_target = None):
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        """Load the data from disk."""
        logger.info(f"Starting to load data from {path}.")

        # If we haven't defined what folds to load, load all:
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            data_file = path.join("train.jsonl.gz")
            self.vocab_source = self._build_vocab(
                dataset = [datapoint["graph"]["node_features"] for datapoint in data_file.read_by_file_suffix()],
                vocab_size=self.max_vocab_size
            )
            self.vocab_target = self._build_vocab(
                dataset = [datapoint["Target"] for datapoint in data_file.read_by_file_suffix()],
                vocab_size=self.max_vocab_size
            )
            self._loaded_data[DataFold.TRAIN] = self.__load_data(data_file)
            logger.debug("Done loading training data.")
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("valid.jsonl.gz"))
            logger.debug("Done loading validation data.")
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path.join("test.jsonl.gz"))
            logger.debug("Done loading test data.")

    def __load_data(self, data_file: RichPath) -> List[GraphSampleType]:
        return [
            self._process_raw_datapoint(datapoint) for datapoint in data_file.read_by_file_suffix()
        ]

    def _process_raw_datapoint(self, datapoint: Dict[str, Any]) -> GraphWithTargetComment:
        node_features = self._tensorise_node_features(self.vocab_source, datapoint["graph"]["node_features"])
        source_seq = self._tensorise_target_token_sequence(self.vocab_source, self.max_seq_length, datapoint["graph"]["node_features"][:datapoint["Source_len"]])

        type_to_adj_list, type_to_num_incoming_edges = self._process_raw_adjacency_lists(
            raw_adjacency_lists=datapoint["graph"]["adjacency_lists"],
            num_nodes=len(node_features),
        )

        target_value = self._tensorise_target_token_sequence(self.vocab_target, self.max_seq_length, datapoint["Target"])

        return GraphWithTargetComment(
            adjacency_lists=type_to_adj_list,
            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
            node_features=node_features,
            target_value=target_value,
            source_len=datapoint["Source_len"] if "Source_len" in datapoint else 0,
            source_seq=source_seq
        )

    def _build_vocab(
        self, dataset, vocab_size: int, max_num_files: Optional[int] = None
    ) -> Vocabulary:
        """
        Compute model metadata such as a vocabulary.

        Args:
            data: Dataset of method code and comments.
            source_or_taget: 'source' for methods source, 'target' for methods comments.
            vocab_size: Maximal size of the vocabulary to create.
            max_num_files: Maximal number of files to load.
        """
      
        vocab = Vocabulary(add_unk=True, add_pad=True)
        # Make sure to include the START_SYMBOL in the vocabulary as well:
        vocab.add_or_get_id(START_SYMBOL)
        vocab.add_or_get_id(END_SYMBOL)
        cnt = collections.Counter()

        for token_seq in dataset:
            for token in token_seq:
                cnt[token] += 1

        for token, _ in cnt.most_common(vocab_size):
            vocab.add_or_get_id(token)

        return vocab

    def _tensorise_target_token_sequence(
        self, vocab: Vocabulary, length: int, token_seq: Iterable[str],
    ) -> List[int]:
        """
        Tensorise a single example.

        Args:
            vocab: Vocabulary to use for mapping tokens to integer IDs
            length: Length to truncate/pad sequences to.
            token_seq: Sequence of tokens to tensorise.

        Returns:
            List with length elements that are integer IDs of tokens in our vocab.
        """
        tensorised = []

        for i in range(length):
            if i==0:
                tensorised.append(vocab.get_id_or_unk(START_SYMBOL))
            elif len(token_seq) >= i:
                tensorised.append(vocab.get_id_or_unk(token_seq[i-1]))
            elif i == len(token_seq) + 1:
                tensorised.append(vocab.get_id_or_unk(END_SYMBOL))
            else:
                tensorised.append(vocab.get_id_or_unk(vocab.get_pad()))

        return tensorised

    def _tensorise_node_features(
        self, vocab: Vocabulary, token_seq
    ) -> np.ndarray:
        tensorised = []
        for token in token_seq:
            tensorised.append([vocab.get_id_or_unk(token)])

        return tensorised

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["target_value"] = []
        new_batch["source_len"] = []
        new_batch["graph_to_num_nodes"] = []
        new_batch["source_seq"] = []
        return new_batch

    def _add_graph_to_batch(
        self, raw_batch: Dict[str, Any], graph_sample: GraphWithTargetComment
    ) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["target_value"].append(graph_sample.target_value)                
        raw_batch["source_len"].append(graph_sample.source_len)
        raw_batch["graph_to_num_nodes"].append(len(graph_sample.node_features))
        raw_batch["source_seq"].append(graph_sample.source_seq)


    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        return {**batch_features, 
            "source_len": raw_batch["source_len"], 
            "source_seq": raw_batch["source_seq"],
            "graph_to_num_nodes": np.array(raw_batch["graph_to_num_nodes"])
            }, {"target_value": raw_batch["target_value"]}
                               

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types={**data_description.batch_features_types, "source_len": tf.int32, "source_seq": tf.int32,  "graph_to_num_nodes": tf.int32},
            batch_features_shapes={**data_description.batch_features_shapes, "source_len": (None,), "source_seq": (None,50), "graph_to_num_nodes": (None,)},
            batch_labels_types={**data_description.batch_labels_types, "target_value": tf.int32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "target_value": (None, None)},
        )
