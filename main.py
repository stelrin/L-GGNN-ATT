from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from preprocessing.prime_dataset import get_dataset_metadata
from preprocessing.graph_formulation import get_dataset
from utils.dataset_metrics import sparse_to_dense_target
from utils.mean_reciprocal_rank import MeanReciprocalRank
from model import Model


# Model parameters
DATASET_NAME = "test"
BATCH_SIZE = 100
PROPAGATION_STEPS = 100
HIDDEN_SIZE = 100

# Optimizer parameters
INITIAL_LEARNING_RATE = 0.001
DECAY_STEP = 3
DECAY_RATE = 0.1

# Regularization parameters
L2_PENALTY = 10**-5

# Metrics parameters
# TODO: Do not forget to set these to 20 when done debugging
PRECISION_TOP_K = 20
RECALL_TOP_K = 20
MRR_TOP_K = 20


# TODO: Should this function only handle the training loop or should it also handle the testing loop?
def training_loop(model: Model, dataset: tf.data.Dataset, optimizer: tf.optimizers.Optimizer, train: bool = False):
    losses = []

    for (adj_in, adj_out, sequence_of_indexes, session_items, mask, target) in tqdm(dataset):
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

            if train:
                # Applying L2 regularization on all trainable variables
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]) * L2_PENALTY
                batch_loss += l2_loss

                gradients = tape.gradient(batch_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(batch_loss.numpy())

    return losses


def testing_loop(model: Model, dataset: tf.data.Dataset):

    metric_precision = tf.keras.metrics.Precision(top_k=PRECISION_TOP_K)
    metric_recall = tf.keras.metrics.Recall(top_k=RECALL_TOP_K)
    metric_mrr = MeanReciprocalRank(top_k=MRR_TOP_K)

    for (adj_in, adj_out, sequence_of_indexes, session_items, mask, target) in tqdm(dataset):
        # (batch_size, item_count - 1)
        logits = model(
            adj_in=adj_in,
            adj_out=adj_out,
            sequence_of_indexes=sequence_of_indexes,
            session_items=session_items,
            mask=mask,
            target=target,
        )

        # (batch_size, item_count - 1)
        softmax_logits = tf.nn.softmax(logits)

        # (batch_size, item_count - 1)
        dense_target = sparse_to_dense_target(target, tf.shape(softmax_logits)[1])

        metric_precision.update_state(y_pred=softmax_logits, y_true=dense_target)  # Precision@20
        metric_recall.update_state(y_pred=softmax_logits, y_true=dense_target)  # Recall@20
        metric_mrr.update_state(y_pred=softmax_logits, y_true=dense_target)  # MRR@20

    precision = metric_precision.result().numpy()
    recall = metric_recall.result().numpy()
    mrr = metric_mrr.result().numpy()

    return precision, recall, mrr


def main():
    dataset_metadata = get_dataset_metadata(DATASET_NAME)

    train_dataset = get_dataset(dataset_metadata, batch_size=BATCH_SIZE, train=True)
    test_dataset = get_dataset(dataset_metadata, batch_size=BATCH_SIZE, train=False)

    model = Model(
        number_of_nodes=dataset_metadata.item_count,
        propagation_steps=PROPAGATION_STEPS,
        hidden_size=HIDDEN_SIZE,
    )

    # Initial learning rate 0.001 decaying by 0.1 every 3 epochs
    DECAY = DECAY_STEP * (dataset_metadata.train_dataset_size / BATCH_SIZE)  # Decay every 3 epochs relative to the batch size
    decaying_learning_rate = keras.optimizers.schedules.ExponentialDecay(INITIAL_LEARNING_RATE, DECAY, DECAY_RATE, staircase=True)
    optimizer = keras.optimizers.Adam(decaying_learning_rate)

    print("| Training {DATASET_NAME} dataset |")
    losses = training_loop(model, train_dataset, optimizer, train=True)
    # TODO: Plot the losses

    print("| Testing {DATASET_NAME} dataset |")
    precision, recall, mrr = testing_loop(model, test_dataset)

    print("Precision@20: {:.2f}%".format(precision * 100.0))
    print("Recall@20: {:.2f}%".format(recall * 100.0))
    print("MRR@20: {:.2f}%".format(mrr * 100.0))


if __name__ == "__main__":
    main()
