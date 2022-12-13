import tensorflow as tf


class MeanReciprocalRank(tf.keras.metrics.Metric):
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.reset_state()

    def reset_state(self):
        self.sample_rr_accumulator = 0.0
        self.number_of_top_k_batches = 0

    def update_state(self, y_pred: tf.Tensor, y_true: tf.Tensor):
        number_of_classes = tf.shape(y_pred)[1]
        y_pred = y_pred[:self.top_k]
        y_true = y_true[:self.top_k]

        sorted_indices = tf.argsort(y_pred, axis=1, direction="DESCENDING", stable=True)
        _sorted_indices = [[[k, i.numpy()] for i in sorted_indices[k]] for k in range(self.top_k)]
        permutated_true = tf.gather_nd(y_true, indices=_sorted_indices)
        column_reciprocal_rank = tf.cast(tf.stack([range(1, number_of_classes + 1) for k in range(self.top_k)], axis=0), dtype=tf.float32)
        column_reciprocal_rank = tf.math.reciprocal(column_reciprocal_rank)

        mrr = tf.reduce_sum(tf.multiply(permutated_true, column_reciprocal_rank), axis=1)
        mrr = tf.reduce_mean(mrr, axis=0)
        self.sample_rr_accumulator += mrr.numpy()
        self.number_of_top_k_batches += 1

    def result(self):
        return self.sample_rr_accumulator / self.number_of_top_k_batches
