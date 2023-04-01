import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from preprocessing.prime_dataset import DatasetMetadata, get_dataset_metadata
from preprocessing.graph_formulation import get_dataset
from model import Model


# Model parameters
DATASET_NAME = "diginetica"
LOSSLESS = True
EPOCHS = 30
BATCH_SIZE = 100
PROPAGATION_STEPS = 1
HIDDEN_SIZE = 100

# Optimizer parameters
INITIAL_LEARNING_RATE = 0.001
DECAY_STEP = 3
DECAY_RATE = 0.1

# Regularization parameters
L2_PENALTY = 10**-5

# Metrics parameters
PRECISION_TOP_K = 20
RECALL_TOP_K = 20
MRR_TOP_K = 20


def training_loop(model: Model, dataset: tf.data.Dataset, optimizer: tf.optimizers.Optimizer, dataset_metadata: DatasetMetadata):
    losses = []

    with tqdm(total=np.floor(dataset_metadata.train_dataset_size / BATCH_SIZE) + 1) as pbar:
        for (adj_in, adj_out, sequence_of_indexes, session_items, mask, target) in dataset:
            pbar.update(1)
            
            with tf.GradientTape() as tape:
                # (batch_size, item_count - 1)
                logits = model(
                    adj_in=adj_in,
                    adj_out=adj_out,
                    sequence_of_indexes=sequence_of_indexes,
                    session_items=session_items,
                    mask=mask,
                    target=target,
                )

                # (batch_size,)
                # We subtract 1 of the targets because the first element is not a real item, it's just there because item ids start at 1
                batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target - 1, logits=logits)

                # (None,)
                batch_loss = tf.reduce_mean(batch_loss)

                # Applying L2 regularization on all trainable variables
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]) * L2_PENALTY
                batch_loss += l2_loss

                gradients = tape.gradient(batch_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            losses.append(batch_loss.numpy())

    return losses


def testing_loop(model: Model, dataset: tf.data.Dataset, dataset_metadata: DatasetMetadata):

    hit, mrr = [], []

    with tqdm(total=np.floor(dataset_metadata.test_dataset_size / BATCH_SIZE) + 1) as pbar:
        for (adj_in, adj_out, sequence_of_indexes, session_items, mask, target) in dataset:
            pbar.update(1)

            # (batch_size, item_count - 1)
            logits = model(
                adj_in=adj_in,
                adj_out=adj_out,
                sequence_of_indexes=sequence_of_indexes,
                session_items=session_items,
                mask=mask,
                target=target,
            )

            index = np.argsort(logits, 1)[:, -MRR_TOP_K:]
            for score, target in zip(index, target):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (MRR_TOP_K - np.where(score == target - 1)[0][0]))
        
        hit = np.array(hit).mean()
        mrr = np.array(mrr).mean()

    return hit, mrr


def main():
    dataset_metadata = get_dataset_metadata(DATASET_NAME)

    train_dataset = get_dataset(dataset_metadata, batch_size=BATCH_SIZE, train=True, lossless=LOSSLESS)
    test_dataset = get_dataset(dataset_metadata, batch_size=BATCH_SIZE, train=False, lossless=LOSSLESS)

    model = Model(
        number_of_nodes=dataset_metadata.item_count,
        propagation_steps=PROPAGATION_STEPS,
        hidden_size=HIDDEN_SIZE,
    )

    # Initial learning rate decaying by 0.1 every 3 epochs
    DECAY = DECAY_STEP * (dataset_metadata.train_dataset_size / BATCH_SIZE)  # Decay every 3 epochs relative to the batch size
    decaying_learning_rate = keras.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE, DECAY, DECAY_RATE, staircase=True)
    optimizer = keras.optimizers.Adam(decaying_learning_rate)

    # Checkpointing
    checkpoint = tf.train.Checkpoint(model)

    losses = []

    for epoch in range(EPOCHS):
        print("------------------")
        losses = training_loop(model, train_dataset, optimizer, dataset_metadata)
        print("Epoch: {} - Loss: {:.4f}".format(epoch, np.mean(losses)))
        print("------------------")

        checkpoint.save('./checkpoints/training_checkpoints')

    # Load the latest checkpoint
    # checkpoint.restore("./checkpoints/training_checkpoints-30")    

    hit, mrr = testing_loop(model, test_dataset, dataset_metadata)
    print("Hit@20: {:.6f} - MRR@20: {:.6f}".format(hit, mrr))


if __name__ == "__main__":
    main()
