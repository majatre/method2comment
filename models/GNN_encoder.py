"""General task for graph binary classification."""
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from tf2_gnn import GNNInput, GNN
from tf2_gnn.data import GraphDataset
from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput


class GraphEncoder(tf.keras.Model):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        """Get the default hyperparameter dictionary for the class."""
        params = {f"gnn_{name}": value for name, value in GNN.get_default_hyperparameters(mp_style).items()}
        these_hypers: Dict[str, Any] = {
            "graph_aggregation_size": 256,
            "graph_aggregation_num_heads": 16,
            "graph_aggregation_hidden_layers": [128],
            "graph_aggregation_dropout_rate": 0.2,
            "token_embedding_size":  64,
            "gnn_message_calculation_class": "gnn_edge_mlp",
            "gnn_hidden_dim": 64,
            "gnn_global_exchange_mode": "mlp",
            "gnn_num_layers": 8,
            "graph_encoding_size": 128,
        }
        params.update(these_hypers)
        return params

    def __init__(self, params: Dict[str, Any], vocab_size, name: str = None):
        super().__init__(name=name)
        self._params = params
        self._num_edge_types = 1
        self._token_embedding_size = params["token_embedding_size"]
        self.vocab_size = vocab_size
            

    def build(self, input_shapes: Dict[str, Any]):
        graph_params = {
            name[4:]: value for name, value in self._params.items() if name.startswith("gnn_")
        }
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self._params["token_embedding_size"])
        self._gnn = GNN(graph_params)
        self._gnn.build(
            GNNInput(
                node_features=self.get_initial_node_feature_shape(input_shapes),
                adjacency_lists=tuple(
                    input_shapes[f"adjacency_list_{edge_type_idx}"]
                    for edge_type_idx in range(self._num_edge_types)
                ),
                node_to_graph_map=tf.TensorShape((None,)),
                num_graphs=tf.TensorShape(()),
            )
        )

        with tf.name_scope(self._name):
          self._node_to_graph_repr_layer = WeightedSumGraphRepresentation(
              graph_representation_size=self._params["graph_aggregation_size"],
              num_heads=self._params["graph_aggregation_num_heads"],
              scoring_mlp_layers=self._params["graph_aggregation_hidden_layers"],
              scoring_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
              transformation_mlp_layers=self._params["graph_aggregation_hidden_layers"],
              transformation_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
          )
          self._node_to_graph_repr_layer.build(
              NodesToGraphRepresentationInput(
                  node_embeddings=tf.TensorShape(
                      (None, input_shapes["node_features"][-1] + self._params["gnn_hidden_dim"])
                  ),
                  node_to_graph_map=tf.TensorShape((None)),
                  num_graphs=tf.TensorShape(()),
              )
          )

          self._graph_repr_layer = tf.keras.layers.Dense(
              self._params["graph_encoding_size"], use_bias=True
          )
          self._graph_repr_layer.build(
              tf.TensorShape((None, self._params["graph_aggregation_size"]))
          )
        super().build([])

    def get_initial_node_feature_shape(self, input_shapes) -> tf.TensorShape:
        return (None, self._token_embedding_size)

    def compute_initial_node_features(self, inputs, training: bool) -> tf.Tensor:
        return tf.squeeze(self.embedding(inputs["node_features"]))

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: tf.Tensor,
        training: bool,
    ) -> Any:
      per_graph_results = self._node_to_graph_repr_layer(
        NodesToGraphRepresentationInput(
          node_embeddings=tf.concat(
              [batch_features["node_features"], final_node_representations], axis=-1
          ),
          node_to_graph_map=batch_features["node_to_graph_map"],
          num_graphs=batch_features["num_graphs_in_batch"],
        )
      )  # Shape [G, graph_aggregation_num_heads]
      per_graph_results = self._graph_repr_layer(
          per_graph_results
      )  # Shape [G, graph_encoding_size]

      return per_graph_results

    def call(self, inputs, training: bool):
        # Pack input data from keys back into a tuple:
        adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            inputs[f"adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )

        # Start the model computations:
        initial_node_features = self.compute_initial_node_features(inputs, training)
        gnn_input = GNNInput(
            node_features=initial_node_features,
            adjacency_lists=adjacency_lists,
            node_to_graph_map=inputs["node_to_graph_map"],
            num_graphs=inputs["num_graphs_in_batch"],
        )
        final_node_representations = self._gnn(gnn_input, training)
        return self.compute_task_output(inputs, final_node_representations, training), final_node_representations