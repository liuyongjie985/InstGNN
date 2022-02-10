import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from models.definitions.InstGNN_4 import InstGNN
from utils.data_loading import load_sogou_sparse_graph_data
from utils.constants import *
import utils.utils as utils
from utils.utils import get_instgnn4_main_loop


def train_gat_cora(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    # node_feature B N 27
    # node_labels B N
    # edge_index B N * N

    _, train_node_features, train_node_labels, train_edge_features, train_edge_connections, train_sparse_edge_features, train_sparse_edge_indexs, train_sparse_edge_labels = load_sogou_sparse_graph_data(
        config,
        "train",
        device)
    _, valid_node_features, valid_node_labels, valid_edge_features, valid_edge_connections, valid_sparse_edge_features, valid_sparse_edge_indexs, valid_sparse_edge_labels = load_sogou_sparse_graph_data(
        config,
        "valid",
        device)

    assert len(train_node_features) == len(train_node_labels) == len(train_edge_features) == len(
        train_edge_connections)
    assert len(valid_node_features) == len(valid_node_labels) == len(valid_edge_features) == len(
        valid_edge_connections)
    print("num of train data", len(train_node_features))
    print("num of test data", len(valid_node_features))

    # Step 2: prepare the model
    instgnn = InstGNN(
        num_of_layers=config['num_of_layers'],
        num_of_joint_learning_layers=config['num_of_joint_learning_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        jll_num_heads_per_layer=config['jll_num_heads_per_layer'],
        jll_num_features_per_layer=config['jll_num_features_per_layer'],
        num_edge_features_per_layer=config['num_edge_features_per_layer'],
        jll_edge_num_features=config['jll_edge_num_features'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    mse_loss_fn = nn.MSELoss()
    optimizer = Adam(instgnn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_instgnn4_main_loop(
        config,
        instgnn,
        loss_fn,
        mse_loss_fn,
        optimizer, device)

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        train_graph_data = (
            train_node_features, train_node_labels, train_edge_features, train_edge_connections,
            train_sparse_edge_features, train_sparse_edge_indexs, train_sparse_edge_labels)
        _, _, _, _, _ = main_loop(LoopPhase.TRAIN, train_graph_data, epoch=epoch)
        # Validation loop
        with torch.no_grad():
            try:
                print("validation starting...")
                valid_graph_data = (
                    valid_node_features, valid_node_labels, valid_edge_features, valid_edge_connections,
                    valid_sparse_edge_features, valid_sparse_edge_indexs, valid_sparse_edge_labels)
                time_start = time.time()
                _, _, loss, node_accuracy, edge_accuracy = main_loop(LoopPhase.VAL, valid_graph_data, epoch=epoch)
                print("GAT validation: train_loss=" + str(loss)[:8] + " |epoch=" + str(
                    epoch + 1) + "|valid node acc=" + str(node_accuracy) + "|valid edge acc=" + str(
                    edge_accuracy))
                accuracy = node_accuracy + edge_accuracy
                # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
                # or the val loss keeps going down we won't stop
                if accuracy > BEST_VAL_PERF or loss < BEST_VAL_LOSS:
                    BEST_VAL_PERF = max(accuracy, BEST_VAL_PERF)  # keep track of the best validation accuracy so far
                    BEST_VAL_LOSS = min(loss, BEST_VAL_LOSS)  # and the minimal loss
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
                    # Save model checkpoint
                    ckpt_model_name = f'{config["exp_name"]}_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                    config['test_perf'] = -1
                    torch.save(utils.get_training_state(config, instgnn),
                               os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
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
        utils.get_training_state(config, instgnn),
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
    parser.add_argument("--exp_name", type=str,
                        help="the name of this experiment")
    args = parser.parse_args()

    # Model architecture related

    instgnn_config = {
        "num_of_layers": 3,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 8, 8],
        "num_features_per_layer": [SOGOU_NUM_INPUT_FEATURES, 32, 32, 32],
        "num_edge_features_per_layer": SOGOU_NUM_INPUT_EDGE_FEATURES,
        "num_of_joint_learning_layers": 3,
        # jll_num_heads_per_layer node的层multi-head 长度为层数+1
        "jll_num_heads_per_layer": [8, 8, 8, 1],
        # jll_num_features_per_layer node的层node feature dim 长度为层数+1
        "jll_num_features_per_layer": [32, 32, 32, SOGOU_NUM_CLASSES],
        # jll_edge_num_features edge的层edge feature dim 长度为层数+1
        "jll_edge_num_features": [SOGOU_NUM_INPUT_EDGE_FEATURES, 35, 35, SOGOU_EDGE_NUM_CLASS],
        "add_skip_connection": True,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.1,  # result is sensitive to dropout
        "layer_type": LayerType.IMP2  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(instgnn_config)
    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    train_gat_cora(get_training_args())
