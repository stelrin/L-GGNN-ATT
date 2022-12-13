import tensorflow as tf


def sparse_to_dense_target(target: tf.Tensor, item_count: int):
    # We subtract 1 of the targets because the first element is not a real item, it's just there because item ids start at 1
    target = target - 1
    # Sparse to dense classification targets
    sparse_target = tf.sparse.SparseTensor(
        values=tf.ones_like(target),
        indices=[[i, target[i]] for i in range(len(target))],
        dense_shape=[tf.shape(target)[0], item_count],  # (batch_size, item_count - 1)
    )

    dense_target = tf.sparse.to_dense(sparse_target)

    return dense_target
