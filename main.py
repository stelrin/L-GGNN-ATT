import tensorflow as tf
from tensorflow import keras

from preprocessing.prime_dataset import get_dataset_metadata
from preprocessing.graph_formulation import DatasetMetadata, get_dataset
from model import Model

# Model parameters
DATASET_NAME = "test"
BATCH_SIZE = 5
PROPAGATION_SIZE = 100
HIDDEN_SIZE = 100

# Optimizer parameters
INITIAL_LEARNING_RATE = 0.001
DECAY_STEP = 3
DECAY_RATE = 0.1


def training_loop(model: Model, dataset: tf.data.Dataset, optimizer: tf.optimizers.Optimizer, train: bool = False):
    losses = []

    for (adj_in, adj_out, sequence_of_indexes, session_items, mask, target) in dataset:
        with tf.GradientTape() as tape:
            logits = model(
                adj_in=adj_in,
                adj_out=adj_out,
                sequence_of_indexes=sequence_of_indexes,
                session_items=session_items,
                mask=mask,
                target=target,
            )

            batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target - 1, logits=logits)

        if train:
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(batch_loss)

    return losses


def main():
    item_count, max_sequence_len, max_number_of_nodes, train_dataset_len, test_dataset_len = get_dataset_metadata(DATASET_NAME)

    print(item_count)  # How many items there are in the dataset, +1 for that 0-indexed item that we discard
    print(max_sequence_len)  # The length of the longest session in the train/test dataset
    print(max_number_of_nodes)  # The maximum number of unique items there is in a session in the dataset
    print(train_dataset_len)  # The size of the train dataset
    print(test_dataset_len)  # The size of the test dataset

    dataset_metadata = DatasetMetadata(
        name="test",
        max_number_of_nodes=max_number_of_nodes,
        max_sequence_len=max_sequence_len,
        item_count=item_count,
    )

    train_dataset = get_dataset(dataset_metadata, batch_size=BATCH_SIZE, train=True)
    test_dataset = get_dataset(dataset_metadata, batch_size=BATCH_SIZE, train=False)

    model = Model(number_of_nodes=item_count, propagation_steps=PROPAGATION_SIZE, hidden_size=HIDDEN_SIZE)

    # Initial learning rate 0.001 decaying by 0.1 every 3 epochs
    DECAY = DECAY_STEP * (train_dataset_len / BATCH_SIZE)  # Decay every 3 epochs relative to the batch size
    decaying_learning_rate = keras.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE, DECAY, DECAY_RATE, staircase=True)
    optimizer = keras.optimizers.Adam(decaying_learning_rate)
    
    # FIXME: L2 penalty 10^⁻5. This is not implemented yet!
    unpenaltied_losses = training_loop(model, train_dataset, optimizer, train=True)
    
    print(unpenaltied_losses)

    # NOTE: Calculate performance metrics (P@20, MRR@20)


if __name__ == "__main__":
    main()
