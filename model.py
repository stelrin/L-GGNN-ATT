import math 
import tensorflow as tf
from tensorflow import keras


class Model(keras.Model):
    def __init__(self, number_of_nodes: int, propagation_steps: int, hidden_size: int):
        super(Model, self).__init(name="sr_gnn")
        
        self.number_of_nodes = number_of_nodes
        self.propagation_steps = propagation_steps
        self.hidden_size = hidden_size

        self.standard_deviation = 1.0 / math.sqrt(self.hidden_size)

        # The representations of all the nodes in the dataset
        self.node_representations = None

        # TODO: Declare model layers here
        # GGNN(propagation_steps, hidden_size, number_of_nodes, standard_deviation)
        # soft_attention(hidden_size, standard_deviation)
        # linear_transformation(hidden_size, standard_deviation)

    def call(self, items, adj_in, adj_out, mask, sequence_of_indexes, target):
        # TODO: Compute batch_size from items' 0-axis size
        self.batch_size = tf.shape(items)[0]

        # GGNN(node_representations, item, adj_in, adj_out, batch_size) -> final_state
        # soft_attention(mask, sequence_of_indexes, batch_size, final_state, node_representations) -> session_graph_representation, session_local_representation
        # linear_transformation(session_graph_representation, session_local_representation) -> session_representation
        
        # Softmax classification

        # Returns classification logits, loss is computed in the training loop
        pass