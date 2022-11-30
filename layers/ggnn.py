import tensorflow as tf
from tensorflow import keras


class GGNN(keras.layers.Layer):
    def __init__(
        self,
        hidden_size: tf.int32,
        propagation_steps: tf.int32,
        standard_deviation: tf.float32,
    ):
        super(GGNN, self).__init__(name="ggnn_layer")

        self.hidden_size = hidden_size
        self.propagation_steps = propagation_steps
        self.standard_deviation = standard_deviation

        self.init_weights()

    def init_weights(self):
        init_H_in = tf.random.uniform(
            shape=[self.hidden_size, self.hidden_size],
            minval=-self.standard_deviation,
            maxval=self.standard_deviation,
        )
        init_H_out = tf.random.uniform(
            shape=[self.hidden_size, self.hidden_size],
            minval=-self.standard_deviation,
            maxval=self.standard_deviation,
        )

        init_b_in = tf.random.uniform(
            shape=[self.hidden_size],
            minval=-self.standard_deviation,
            maxval=self.standard_deviation,
        )
        init_b_out = tf.random.uniform(
            shape=[self.hidden_size],
            minval=-self.standard_deviation,
            maxval=self.standard_deviation,
        )

        self.H_in = tf.Variable(init_H_in, name="H_in", dtype=tf.float32)
        self.H_out = tf.Variable(init_H_out, name="H_out", dtype=tf.float32)

        self.b_in = tf.Variable(init_b_in, name="b_in", dtype=tf.float32)
        self.b_out = tf.Variable(init_b_out, name="b_out", dtype=tf.float32)

    def call(
        self,
        node_representations: tf.Tensor,
        session_items: tf.Tensor,
        adj_in: tf.Tensor,
        adj_out: tf.Tensor,
        batch_size: tf.int32,
    ):
        # (batch_size, max_sequence_len, hidden_size)
        node_vectors = tf.nn.embedding_lookup(node_representations, session_items)

        # TODO: Maybe move this cast to the model's call() prior to the GGNN layer's call
        adj_in = tf.cast(adj_in, dtype=tf.float32)
        adj_out = tf.cast(adj_out, dtype=tf.float32)

        recurrent_cell = keras.layers.GRUCell(self.hidden_size)

        for step in range(self.propagation_steps):
            # (batch_size * max_sequence_len, hidden_size)
            _node_vectors = tf.reshape(node_vectors, (-1, self.hidden_size))

            # (batch_size * max_sequence_len, hidden_size)
            node_vectors_in = tf.matmul(_node_vectors, self.H_in) + self.b_in
            node_vectors_out = tf.matmul(_node_vectors, self.H_out) + self.b_out

            # (batch_size, max_sequence_len, hidden_size)
            node_vectors_in = tf.reshape(
                node_vectors_in, (batch_size, -1, self.hidden_size)
            )
            node_vectors_out = tf.reshape(
                node_vectors_out, (batch_size, -1, self.hidden_size)
            )

            # (batch_size, max_sequence_len, hidden_size)
            a_in = tf.matmul(adj_in, node_vectors_in)
            a_out = tf.matmul(adj_out, node_vectors_out)

            # (batch_size, max_sequence_len, hidden_size * 2)
            a = tf.concat([a_in, a_out], axis=-1)

            # (batch_size * max_sequence_len, hidden_size * 2)
            a = tf.reshape(a, (-1, self.hidden_size * 2))

            # TODO: Update RNN implementation to Keras API
            # (batch_size * max_sequence_len, hidden_size)
            _, _node_vectors = tf.compat.v1.nn.dynamic_rnn(
                cell=recurrent_cell,
                inputs=tf.expand_dims(a, axis=1),
                initial_state=_node_vectors,
            )

            # (batch_size, max_sequence_len, hidden_size)
            _node_vectors = tf.reshape(
                _node_vectors, (batch_size, -1, self.hidden_size)
            )

        return _node_vectors
