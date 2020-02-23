import os
from glob import iglob
from typing import List, Dict, Any, Iterable, Optional, Iterator
import re

import numpy as np
import collections
from more_itertools import chunked
from dpu_utils.mlutils.vocabulary import Vocabulary

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge

# import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

DATA_FILE_EXTENSION = "proto"
START_SYMBOL = "%START%"
END_SYMBOL = "%END%"


def get_data_files_from_directory(
    data_dir: str, max_num_files: Optional[int] = None
) -> List[str]:
    files = iglob(
        os.path.join(data_dir, "**/*.%s" % DATA_FILE_EXTENSION), recursive=True
    )
    if max_num_files:
        files = sorted(files)[: int(max_num_files)]
    else:
        files = list(files)
    return files


def format_comment_to_plain_text(comment: str):
    # Delete the params at the end of JavaDoc.
    if comment.find("* @") > 0:
        comment = comment[:comment.find("* @")]
    # Use regular expression to eliminate special JavaDoc characters
    # and unnecessary whitespaces.
    comment = re.sub(r'[*/]', ' ',  comment)  
    comment = re.sub(r'\s+', ' ',  comment)  
    comment = re.sub(r'<[^>]>', ' ',  comment)  
    comment = re.sub(r'{@\w+\s([^}]*)}', r'\1', comment)
    return comment.strip()

  
def load_data_file_methods(file_path: str):
    """
    Load a single data file, returning methods code and JavaDoc comments.
    """

    methods_code = []
    methods_comments = []
    graphs = []

    g = Graph()
    with open(file_path, "rb") as f:
        g.ParseFromString(f.read())

    # Build a dictionary of nodes indexed by id 
    # by start position and end position
    nodes_dict = {}
    tokens_by_start_pos = {}
    tokens_by_end_pos = {}
    # A list of methods root nodes
    methods = []
    for n in g.node:
        nodes_dict[n.id] = n
        if n.contents == 'METHOD':
            methods.append(n)
        if n.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN):
            tokens_by_start_pos[n.startPosition] = n
            tokens_by_end_pos[n.endPosition] = n
    
    # Build a dictionary of edges indexed by source id
    edges_dict = {}
    for e in g.edge:
        if e.sourceId in edges_dict:
            edges_dict[e.sourceId].append(e)
        else:
            edges_dict[e.sourceId] = [e]

    for m in methods:
        # Start with a node that is a token and starts at the same position 
        # as method's start postion
        nid = tokens_by_start_pos[m.startPosition].id
        tokens = []
        comment = ""

        # Follow the 'next token' edges up to the token finishing at end postion
        while nid != tokens_by_end_pos[m.endPosition].id:
            tokens.append(nodes_dict[nid].contents.lower())
            if nid in edges_dict:
                for e in edges_dict[nid]:
                    if e.type == FeatureEdge.NEXT_TOKEN:
                        nid = e.destinationId

        for n in g.node:
            if n.type == FeatureNode.COMMENT_JAVADOC and m.id == edges_dict[n.id][0].destinationId:
                comment = format_comment_to_plain_text(n.contents)

        # I add only the non-empty methods that have comments.
        # I also ensure that method is not vrtual and has a body starting with '{'. 
        if len(tokens) > 0 and len(comment) > 0 and 'lbrace' in tokens and len(tokens) < 200:
           methods_code.append(tokens)
           methods_comments.append(comment)

           methods_edges, nodes_features = get_method_graph(m, nodes_dict, edges_dict)
           
           graph = {'Target': word_tokenize(comment),
                    'Source_len': len(tokens),
                    'graph': {'node_features': nodes_features,
                              'adjacency_lists': methods_edges}}
           if len(nodes_features) < 300:
                graphs.append(graph)

            # print(graph)

    return methods_code, methods_comments, graphs


def get_method_graph(m, nodes_dict, edges_dict):
    method_nodes_ids = []    
    get_method_nodes_rec(m.id, method_nodes_ids, edges_dict)
    methods_edges = [[] for i in range(3)]

    for node in method_nodes_ids:
        if node in edges_dict:
            for edge in edges_dict[node]:
                if edge.destinationId in method_nodes_ids:
                    if edge.type in [FeatureEdge.ASSOCIATED_TOKEN, FeatureEdge.NEXT_TOKEN, FeatureEdge.AST_CHILD]:
                        methods_edges[edge.type-1].append([node, edge.destinationId])

    method_nodes = {node_id: node for node_id, node in nodes_dict.items() if node_id in
                    method_nodes_ids}

    methods_edges, nodes_features = adjust_edges_ids(methods_edges, method_nodes)
        
    return methods_edges, nodes_features


def adjust_edges_ids(edges, nodes):
    old_id_to_new_id = {}
    i = 0
    nodes_values = sorted(nodes.values(), key=lambda node: node.id)
    new_edges = [[] for edge_type in edges]
    nodes_features = []

    # Set new ids for tokens
    for node_value in nodes_values:
        if node_value.type == FeatureNode.TOKEN or node_value.type == FeatureNode.IDENTIFIER_TOKEN:
            old_id_to_new_id[node_value.id] = i
            nodes_features += [node_value.contents.lower()]
            i += 1

    # Set new ids for other nodes
    for node_value in nodes_values:
        if not (node_value.type == FeatureNode.TOKEN or node_value.type == FeatureNode.IDENTIFIER_TOKEN):
            old_id_to_new_id[node_value.id] = i
            nodes_features += [FeatureNode.NodeType.Name(node_value.type)]
            i += 1

    for i, edge_type in enumerate(edges):
        for edge in edge_type:
            new_edges[i].append((old_id_to_new_id[edge[0]], old_id_to_new_id[edge[1]]))

    return new_edges, nodes_features


def get_method_nodes_rec(node_id, method_nodes_ids, edges_dict):
    """
    Utilities to recursively retrieve all edges of a method graph.
    """
    method_nodes_ids.append(node_id)

    if node_id in edges_dict:
        for edge in edges_dict[node_id]:
            if edge.type != FeatureEdge.NEXT_TOKEN and edge.destinationId not in method_nodes_ids:
                get_method_nodes_rec(edge.destinationId, method_nodes_ids, edges_dict)


def generate_dataset_from_dir(data_dir):
    """
    Extract method source code and comments from given directory.

    Args:
        data_dir: Directory from which to load the data.
        max_num_files: Number of files to load at most.

    Returns:
        (methods_code, methods_comment)
    """
    methods_code = []
    methods_comments = []
    graphs = []

    data_files = get_data_files_from_directory(data_dir)

    i = 0
    for data_file in data_files:
        file_methods_code, file_methods_comments, file_graphs = load_data_file_methods(data_file)
        methods_code += file_methods_code
        methods_comments += file_methods_comments
        graphs += file_graphs

        if i%10 == 0:
            print(f"Processed {i}/{len(data_files)} files, added {len(methods_code)} methods in total.")
        i += 1

    return methods_code, methods_comments, graphs


def build_vocab(
    data: dict, source_or_target: str, vocab_size: int, max_num_files: Optional[int] = None
) -> Vocabulary:
    """
    Compute model metadata such as a vocabulary.

    Args:
        data: Dataset of method code and comments.
        source_or_taget: 'source' for methods source, 'target' for methods comments.
        vocab_size: Maximal size of the vocabulary to create.
        max_num_files: Maximal number of files to load.
    """
    if source_or_target == "source":
        dataset = data['methods_code']
    else:
        dataset = data['methods_comments']

    vocab = Vocabulary(add_unk=True, add_pad=True)
    # Make sure to include the START_SYMBOL in the vocabulary as well:
    vocab.add_or_get_id(START_SYMBOL)
    vocab.add_or_get_id(END_SYMBOL)
    cnt = collections.Counter()

    for token_seq in dataset:
        if source_or_target == "target":
            token_seq = word_tokenize(token_seq)
        for token in token_seq:
            cnt[token] += 1

    for token, _ in cnt.most_common(vocab_size):
        vocab.add_or_get_id(token)

    return vocab


def tensorise_token_sequence(
    vocab: Vocabulary, length: int, token_seq: Iterable[str],
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


def tensorise_data(
    vocab: Vocabulary, length: int, data: dict, source_or_target: str, max_num_files: Optional[int] = None
) -> np.ndarray:
    """
    Tensorise data.
    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        data: Dictionary that contains the data.
        source_or_taget: 'source' for methods source, 'target' for methods comments.
        max_num_files: Number of files to load at most.
    Returns:
        numpy int32 array of shape [None, length], containing the tensorised
        data.
    """
    if source_or_target == "source":
        dataset = data['methods_code']
    else:
        dataset = data['methods_comments']

    tensorised_dataset = []
    for token_seq in dataset:
        if source_or_target == "target":
            token_seq = word_tokenize(token_seq)
        tensorised_dataset.append(tensorise_token_sequence(vocab, length, token_seq))

    return np.array(tensorised_dataset, dtype=np.int32)

def prepare_data(
    vocab_source: Vocabulary, vocab_target: Vocabulary, data: dict, 
    max_source_len:int, max_target_len:int, max_num_files: Optional[int] = None
) -> np.ndarray:
    data_source = tensorise_data(
        vocab_source,
        length=max_source_len,
        data=data,
        source_or_target = "source",
        max_num_files=max_num_files,
    )
    data_target = tensorise_data(
        vocab_target,
        length=max_target_len,
        data=data,
        source_or_target = "target",
        max_num_files=max_num_files,
    )
    return np.array(list(zip(data_source, data_target)), dtype=np.int32)

def get_minibatch_iterator(
    token_seqs: np.ndarray,
    batch_size: int,
    is_training: bool,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
    indices = np.arange(token_seqs.shape[0])
    if is_training:
        np.random.shuffle(indices)

    for minibatch_indices in chunked(indices, batch_size):
        if len(minibatch_indices) < batch_size and drop_remainder:
            break  # Drop last, smaller batch

        minibatch_seqs = token_seqs[minibatch_indices]
        yield minibatch_seqs
