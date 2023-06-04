import tensorflow as tf
from tensorflow import keras


class SessionGraphSoftAttention(keras.layers.Layer):
    def __init__(self, hidden_size: tf.int32, standard_deviation: tf.float32):
        super(SessionGraphSoftAttention, self).__init__(name="session_graph_soft_attention")

        self.hidden_size = hidden_size
        self.standard_deviation = standard_deviation

        self.init_weights()

    def init_weights(self):
        # nn_1
        self.linear_last_item = keras.layers.Dense(
            name="linear_last_item",
            units=self.hidden_size,
            input_dim=self.hidden_size,
            activation=None,
            kernel_initializer=tf.initializers.RandomUniform(minval=-self.standard_deviation, maxval=self.standard_deviation),
            use_bias=False,
        )

        # nn_2
        self.linear_sequence_items = keras.layers.Dense(
            name="linear_sequence_items",
            units=self.hidden_size,
            input_dim=self.hidden_size,
            activation=None,
            kernel_initializer=tf.initializers.RandomUniform(minval=-self.standard_deviation, maxval=self.standard_deviation),
            bias_initializer=tf.initializers.Zeros(),
        )

        # nn_3
        self.linear_alpha = keras.layers.Dense(
            name="linear_alpha",
            units=1,
            input_dim=self.hidden_size,
            activation=None,
            kernel_initializer=tf.initializers.RandomUniform(minval=-self.standard_deviation, maxval=self.standard_deviation),
            use_bias=False,
        )


    def get_last_item_representation(self, mask: tf.Tensor, sequence_of_indexes: tf.Tensor, node_vectors: tf.Tensor, batch_size: tf.int32):
        # (batch_size,)
        sequence_length = tf.reduce_sum(mask, axis=1)

        # (batch_size,)
        last_item_index = tf.gather_nd(
            sequence_of_indexes,
            tf.stack(
                [tf.range(batch_size), tf.cast(sequence_length, tf.int32) - 1],
                axis=1,
            ), # Each session's last item index in node_vectors
        )

        # (batch_size, hidden_size)
        last_item_representation = tf.gather_nd(
            node_vectors,
            tf.stack(
                [tf.range(batch_size), last_item_index],
                axis=1)
        )

        return last_item_representation

    def call(self, sequence_of_indexes: tf.Tensor, mask: tf.Tensor, node_vectors: tf.Tensor, batch_size: tf.int32):
        # (batch_size, max_sequence_len)
        mask = tf.cast(mask, tf.float32)

        # (batch_size, hidden_size)
        v_n = self.get_last_item_representation(mask=mask, sequence_of_indexes=sequence_of_indexes, node_vectors=node_vectors, batch_size=batch_size)

        # (batch_size, hidden_size)
        v_n_post_linear = self.linear_last_item(v_n)

        # Shape change to allow for an addition broadcasting later on (cf. sigmoid_result)
        # (batch_size, 1, hidden_size)
        _v_n_post_linear = tf.reshape(v_n_post_linear, [batch_size, 1, self.hidden_size])

        # (batch_size, max_sequence_len, hidden_size)
        v_i = tf.stack(
            [tf.nn.embedding_lookup(node_vectors[i], sequence_of_indexes[i]) for i in range(batch_size)],
            axis=0,
        )

        # (batch_size * max_sequence_len, hidden_size)
        _v_i = tf.reshape(v_i, [-1, self.hidden_size])

        # (batch_size * max_sequence_len, hidden_size)
        v_i_post_linear = self.linear_sequence_items(_v_i)

        # (batch_size, max_sequence_len, hidden_size)
        _v_i_post_linear = tf.reshape(v_i_post_linear, [batch_size, -1, self.hidden_size])

        # We add v_n to each item v_i with addition broadcasting, then we sigmoid the result
        # (batch_size, max_sequence_len, hidden_size)
        sigmoid_result = tf.sigmoid(_v_n_post_linear + _v_i_post_linear)

        # (batch_size * max_sequence_len, hidden_size)
        _sigmoid_result = tf.reshape(sigmoid_result, [-1, self.hidden_size])

        # (batch_size * max_sequence_len, 1)
        alpha = self.linear_alpha(_sigmoid_result)

        # We get rid of the 0-indexed fictional item's weights so it doesn't affect the aggregation
        # (batch_size * max_sequence_len, 1)
        _alpha = alpha * tf.reshape(mask, [-1, 1])

        # We reshape it to calculate the aggregation of the node representations of each node in the session
        # (batch_size, max_sequence_len, 1)
        _alpha = tf.reshape(_alpha, [batch_size, -1, 1])

        # (batch_size, hidden_size)
        session_local_representation = v_n

        # We multiply each item v_i by its alpha with multiplication broadcasting, then we calculate the aggregation of all the items in the sequence
        # (batch_size, hidden_size)
        session_graph_representation = tf.reduce_sum(_alpha * v_i, axis=1)

        return session_local_representation, session_graph_representation
