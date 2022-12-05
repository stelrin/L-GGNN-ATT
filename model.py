import math
import tensorflow as tf
from tensorflow import keras

from layers.ggnn import GGNN
from layers.soft_attention import SessionGraphSoftAttention
from layers.linear_transformation import SessionRepresentationLinearTransformation


class Model(keras.Model):
    def __init__(self, number_of_nodes: tf.int32, propagation_steps: tf.int32, hidden_size: tf.int32):
        super(Model, self).__init__(name="sr_gnn")

        self.number_of_nodes = number_of_nodes
        self.propagation_steps = propagation_steps
        self.hidden_size = hidden_size

        self.standard_deviation = 1.0 / math.sqrt(self.hidden_size)

        init_node_representations = tf.random.uniform(shape=[self.number_of_nodes, self.hidden_size], minval=-self.standard_deviation, maxval=self.standard_deviation)
        self.node_representations = tf.Variable(init_node_representations, name="node_representations", dtype=tf.float32)

        self.ggnn = GGNN(hidden_size=self.hidden_size, propagation_steps=self.propagation_steps, standard_deviation=self.standard_deviation)
        self.soft_attention = SessionGraphSoftAttention(hidden_size=self.hidden_size, standard_deviation=self.standard_deviation)
        self.linear_transformation = SessionRepresentationLinearTransformation(hidden_size=self.hidden_size, standard_deviation=self.standard_deviation)

    def call(self, session_items: tf.Tensor, adj_in: tf.Tensor, adj_out: tf.Tensor, mask: tf.Tensor, sequence_of_indexes: tf.Tensor, target: tf.Tensor):
        batch_size = tf.shape(session_items)[0]

        # (batch_size, max_number_of_nodes, hidden_size)
        node_vectors = self.ggnn(
            node_representations=self.node_representations,
            session_items=session_items,
            adj_in=adj_in,
            adj_out=adj_out,
            batch_size=batch_size,
        )

        # (batch_size, hidden_size), (batch_size, hidden_size)
        session_local_representation, session_graph_representation = self.soft_attention(
            sequence_of_indexes=sequence_of_indexes,
            mask=mask,
            node_vectors=node_vectors,
            batch_size=batch_size,
        )

        # (batch_size, hidden_size)
        session_representation = self.linear_transformation(
            session_local_representation=session_local_representation,
            session_graph_representation=session_graph_representation,
        )

        # The first element is not a real item, it's just there because item ids start at 1
        node_representations = self.node_representations[1:]

        # (batch_size, number_of_nodes)
        # The actual number of items in the dataset, without the added 1
        logits = tf.matmul(session_representation, node_representations, transpose_b=True)

        return logits
