import tensorflow as tf
from tensorflow import keras


class SessionRepresentationLinearTransformation(keras.layers.Layer):
    def __init__(self, hidden_size: tf.Tensor, standard_deviation: tf.float32):
        super(SessionRepresentationLinearTransformation, self).__init__(name="session_representation_linear_transformation")

        self.hidden_size = hidden_size
        self.standard_deviation = standard_deviation

        self.init_weights()

    def init_weights(self):
        standard_deviation_initializer = tf.initializers.RandomUniform(minval=-self.standard_deviation, maxval=self.standard_deviation)

        self.linear_transformation = keras.layers.Dense(activation=None, name="linear_transformation", units=self.hidden_size, kernel_initializer=standard_deviation_initializer, use_bias=False)

        self.linear_transformation.build(self.hidden_size * 2)

    def call(self, session_local_representation: tf.Tensor, session_graph_representation: tf.Tensor):
        # (batch_size, hidden_size * 2)
        concatenation = tf.concat([session_local_representation, session_graph_representation], axis=1)

        # (batch_size, hidden_size)
        session_representation = self.linear_transformation(concatenation)

        return session_representation
