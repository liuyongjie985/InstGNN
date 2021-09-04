import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from models.definitions.GAT import GAT
from utils.data_loading import load_sogou_graph_data
from utils.constants import *
import utils.utils as utils


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat, cross_entropy_loss, optimizer):
    def main_loop(phase, graph_data, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()
        NUM_BATCH = len(graph_data[0])
        total_loss = 0
        total_accuracy = 0
        # time_start = time.time()
        for single_graph_data in zip(*graph_data):
            # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
            # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
            # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
            nodes_unnormalized_scores = gat((single_graph_data[0], single_graph_data[2]))[0]
            # N*1
            gt_node_labels = single_graph_data[1]  # gt stands for ground truth
            # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
            # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
            # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
            # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
            # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
            # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
            # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
            loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)
            total_loss += loss.item()
            if phase == LoopPhase.TRAIN:
                optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                optimizer.step()  # apply the gradients to weights

            # Calculate the main metric - accuracy

            # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
            # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
            class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
            accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)
            total_accuracy += accuracy
            # Logging
            if phase == LoopPhase.TRAIN:
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), epoch)
                    writer.add_scalar('training_acc', accuracy, epoch)

                # Log to console

                # print("GAT training: time elapsed=" + str(time.time() - time_start) + "[s]|epoch=" + str(
                #     epoch + 1) + "|train acc=" + str(accuracy))

            elif phase == LoopPhase.VAL:
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), epoch)
                    writer.add_scalar('val_acc', accuracy, epoch)

        return total_loss / NUM_BATCH, total_accuracy / NUM_BATCH

    return main_loop  # return the decorated function


def train_gat_cora(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    # node_feature B N 27
    # node_labels B N
    # edge_index B N * N
    _, train_node_features, train_node_labels, train_edge_index = load_sogou_graph_data(config, "train", device)
    _, valid_node_features, valid_node_labels, valid_edge_index = load_sogou_graph_data(config, "valid", device)

    assert len(train_node_features) == len(train_node_labels) == len(train_edge_index)
    assert len(valid_node_features) == len(valid_node_labels) == len(valid_edge_index)
    print("num of train data", len(train_node_features))
    print("num of test data", len(valid_node_features))

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer)

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        train_graph_data = (train_node_features, train_node_labels, train_edge_index)
        _, _ = main_loop(LoopPhase.TRAIN, train_graph_data, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                print("validation starting...")
                valid_graph_data = (valid_node_features, valid_node_labels, valid_edge_index)
                time_start = time.time()
                loss, accuracy = main_loop(LoopPhase.VAL, valid_graph_data, epoch=epoch)
                print("GAT validation: time elapsed=" + str(time.time() - time_start) + "[s]|epoch=" + str(
                    epoch + 1) + "|valid acc=" + str(accuracy))
                # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
                # or the val loss keeps going down we won't stop
                if accuracy > BEST_VAL_PERF or loss < BEST_VAL_LOSS:
                    BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                    BEST_VAL_LOSS = min(loss, BEST_VAL_LOSS)  # and the minimal loss
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
                    # Save model checkpoint
                    ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                    config['test_perf'] = -1
                    torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
                else:
                    PATIENCE_CNT += 1  # otherwise keep counting

                if PATIENCE_CNT >= config['patience_period']:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')

            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
    if config['should_test']:
        test_acc = main_loop(phase=LoopPhase.TEST)
        config['test_perf'] = test_acc
        print(f'Test accuracy = {test_acc}')
    else:
        config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    torch.save(
        utils.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['dataset_name']))
    )


def get_training_args():
    parser = argparse.ArgumentParser()
    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int,
                        help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true',
                        help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training',
                        default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)",
                        default=100)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 4,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 8, 8, 1],
        "num_features_per_layer": [SOGOU_NUM_INPUT_FEATURES, 32, 32, 32, SOGOU_NUM_CLASSES],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.1,  # result is sensitive to dropout
        "layer_type": LayerType.IMP2  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    train_gat_cora(get_training_args())
