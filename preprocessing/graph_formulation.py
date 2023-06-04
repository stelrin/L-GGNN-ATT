import csv
import tensorflow as tf
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from preprocessing.prime_dataset import DatasetMetadata

def graph_formulation(row):
    # All the interactions in the sequence except the last one
    features = row[:-1]

    # The last item in the sequence
    target = row[-1]

    # items: All the items that occurred in the features
    # alias_inputs: The features, but for each item we replace it with its id in `items`
    session_items, sequence_of_indexes = tf.unique(features)

    # vector_length: the length of the feature vector
    # n_nodes: number of nodes in the graph
    # indices: the elements of the first row are the indexes of the source nodes
    #          the elements of the second row are the indexes of the destination nodes
    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(session_items)[0]
    indices = tf.gather(
        sequence_of_indexes,
        tf.stack([tf.range(vector_length - 1), tf.range(vector_length - 1) + 1], axis=0),
    )  # Stack and stagger values

    # Grabbing the unique indices
    unique_indices, _ = tf.unique(indices[0] * (vector_length + 1) + indices[1])  # unique(a*x + b)

    unique_indices = tf.sort(unique_indices)  # Sort ascendingly
    unique_indices = tf.stack(
        [
            tf.math.floordiv(unique_indices, (vector_length + 1)),
            tf.math.floormod(unique_indices, (vector_length + 1)),
        ],
        axis=1,
    )  # Ungroup and stack
    unique_indices = tf.cast(unique_indices, tf.int64)

    # values, one vector which size is the number of edges in the current session
    values = tf.ones(tf.shape(unique_indices, out_type=tf.int64)[0], dtype=tf.int64)
    dense_shape = tf.cast([n_nodes, n_nodes], tf.int64)

    # (n_node, n_node) adjacency matrix
    adj = tf.SparseTensor(indices=unique_indices, values=values, dense_shape=dense_shape)
    adj = tf.sparse.to_dense(adj)

    # formulating incoming adjacency matrix
    u_sum_in_tf = tf.math.reduce_sum(adj, 0) # Calculates the incoming edges of each node
    u_sum_in_tf = tf.clip_by_value(u_sum_in_tf, 1, tf.reduce_max(u_sum_in_tf))
    adj_in = tf.math.divide(adj, u_sum_in_tf)

    # formulating outgoing adjacency matrix
    u_sum_out_tf = tf.math.reduce_sum(adj, 1) # Calculates the outgoing edges of each node
    u_sum_out_tf = tf.clip_by_value(u_sum_out_tf, 1, tf.reduce_max(u_sum_out_tf))
    adj_out = tf.math.divide(tf.transpose(adj), u_sum_out_tf)

    # always a one matrix, used to compute the length of the sequence after padding
    mask = tf.fill(tf.shape(features), 1) # NOTE: ones_like does just that

    return adj_in, adj_out, sequence_of_indexes, session_items, mask, target

def graph_formulation_lossless(row):
    features = row[:-1]
    target = row[-1]
    session_items, sequence_of_indexes = tf.unique(features)
    mask = tf.fill(tf.shape(features), 1)

    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(session_items)[0]
    
    order = [0.0 for _ in range(n_nodes)]    
    indices = []
    values = []

    if vector_length == 1:
        adj_in = tf.zeros(shape=[1, 1], dtype=tf.float32)
        adj_out = tf.zeros(shape=[1, 1], dtype=tf.float32)
    else:
        for i in range(vector_length - 1):
            curr = sequence_of_indexes[i].numpy()
            next = sequence_of_indexes[i + 1].numpy()
            
            order[next] += 1.0
            indices.append([curr, next])
            values.append(order[next])

        indices = np.array(indices, dtype=np.int64)
        values = np.array(values, dtype=np.float32)
                
        # @Ali ‒ What's the ideal course of action when edge (i, j) appears multiple times, merge them by summing their weights?

        max_weight = tf.reduce_max(values)
        edge_weights = defaultdict(np.float32)

        for (i, j, weight) in zip(indices[:, 0], indices[:, 1], values):
            edge_weights[(i, j)] += weight

        merged_indices = []
        merged_values = []
        for (i, j), weight in edge_weights.items():
            merged_indices.append([i, j])
            merged_values.append(weight)

        merged_indices = tf.constant(merged_indices, dtype=tf.int64)
        merged_values = tf.constant(merged_values, dtype=tf.float32)

        sparse_adjacency = tf.sparse.SparseTensor(dense_shape=[n_nodes, n_nodes], values=merged_values, indices=merged_indices)
        sparse_adjacency = tf.sparse.reorder(sparse_adjacency)
        
        # Or pick the maximum weight? (add validate_indices=True arg to .to_dense())
        adj_in = tf.sparse.to_dense(sparse_adjacency)
        
        # Normalizing the adjacency matrices
        adj_in = tf.divide(adj_in, max_weight)
        adj_out = tf.transpose(adj_in)
    
    # The adjacency matrices appear to be transposed in the original implementation;
    # This explains some of the unsimilarities between the implementation and the original paper, variable names are misleading.
    return adj_in, adj_out, sequence_of_indexes, session_items, mask, target


def get_dataset(dataset_info: DatasetMetadata, batch_size: int, train: bool = True, lossless=False):
    with open(f"datasets/{dataset_info.name}/{'train' if train else 'test'}.csv", "r") as preprocessed_file:
        data = [list(map(int, rec)) for rec in csv.reader(preprocessed_file, delimiter=",")]

    print("Formulating session sequences into graph structures...")

    if lossless:
        # graph_formulation_lossless is not graph-based which means it can't be used with Dataset.map() to parallelize the transformation
        data = [graph_formulation_lossless(row) for row in tqdm(data)]
    else:
        # Dataset.map() can be used to apply the graph_formulation transformation
        data = [graph_formulation(row) for row in tqdm(data)]
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data,
        output_types=(
            tf.float32,  # adj_in
            tf.float32,  # adj_out
            tf.int32,    # sequence_of_indexes
            tf.int32,    # session_items
            tf.int32,    # mask
            tf.int32,    # target
        )
    )

    print("Padding batches...")

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [dataset_info.max_number_of_nodes, dataset_info.max_number_of_nodes],  # adj_in
            [dataset_info.max_number_of_nodes, dataset_info.max_number_of_nodes],  # adj_out
            [dataset_info.max_sequence_len],  # sequence_of_indexes
            [dataset_info.max_number_of_nodes],  # session_items
            [dataset_info.max_sequence_len],  # mask
            [],  # target
        ),
    )

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
