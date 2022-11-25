import csv
import tensorflow as tf
from functools import partial
from typing import NamedTuple

def generator(data):
    for example in data:
        yield example


def process_data(row):
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
        tf.stack(
            [tf.range(vector_length - 1), tf.range(vector_length - 1) + 1], axis=0
        ),
    )  # Stack and stagger values

    # TODO: Change graph formulation here
    unique_indices, _ = tf.unique(
        indices[0] * (vector_length + 1) + indices[1]
    )  # unique(a*x + b)

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
    adj = tf.SparseTensor(
        indices=unique_indices, values=values, dense_shape=dense_shape
    )
    adj = tf.sparse.to_dense(adj)

    # formulating incoming adjacency matrix
    u_sum_in_tf = tf.math.reduce_sum(adj, 0)
    u_sum_in_tf = tf.clip_by_value(u_sum_in_tf, 1, tf.reduce_max(u_sum_in_tf))
    A_in = tf.math.divide(adj, u_sum_in_tf)

    # formulating outgoing adjacency matrix
    u_sum_out_tf = tf.math.reduce_sum(adj, 1)
    u_sum_out_tf = tf.clip_by_value(u_sum_out_tf, 1, tf.reduce_max(u_sum_out_tf))
    A_out = tf.math.divide(tf.transpose(adj), u_sum_out_tf)

    # always a one matrix, used to compute the length of the sequence after padding
    mask = tf.fill(tf.shape(features), 1)

    return A_in, A_out, sequence_of_indexes, session_items, mask, target

# TODO: Move this to prime_dataset.py
class DatasetMetadata(NamedTuple):
    name: str
    max_sequence_len: int
    item_count: int


# TODO: Store dataset related metadata somewhere else instead of feeding it to get_dataset (maybe a Dataset class would do)
def get_dataset(
    dataset_info: DatasetMetadata,
    batch_size: int,
    train: bool = True,
):
    with open(f"datasets/{dataset_info.name}/{'train' if train else 'test'}.csv", "r") as preprocessed_file:
        data = [
            list(map(int, rec)) for rec in csv.reader(preprocessed_file, delimiter=",")
        ]

    dataset = tf.data.Dataset.from_generator(
        partial(generator, data), output_types=(tf.int32)
    )

    dataset = dataset.map(
        process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # TODO: shuffle should take the size of the train/test dataset
    dataset = dataset.shuffle(100000)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [dataset_info.item_count, dataset_info.item_count],    # A_in
            [dataset_info.item_count, dataset_info.item_count],    # A_out
            [dataset_info.max_sequence_len],                       # sequence_of_indexes
            [dataset_info.item_count],                             # session_items
            [dataset_info.max_sequence_len],                       # mask
            [],                                                    # target
        ),
    )

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
