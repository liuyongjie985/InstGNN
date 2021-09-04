import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from models.definitions.GAT import GAT
from utils.data_loading import load_sogou_graph_data
from utils.constants import *
import json
import matplotlib.pyplot as plt
import traceback


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
        class_pd_list = []
        i = 0
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
            class_pd_list.append(class_predictions)
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
                print(str(i) + "/" + str(NUM_BATCH) + ":" + str(accuracy))
            i += 1
        return class_pd_list, total_loss / NUM_BATCH, total_accuracy / NUM_BATCH

    return main_loop  # return the decorated function


def test_gat_cora(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    # node_feature B N 27
    # node_labels B N
    # edge_index B N * N
    valid_file_name, valid_node_features, valid_node_labels, valid_edge_index = load_sogou_graph_data(config, "valid",
                                                                                                      device)

    assert len(valid_node_features) == len(valid_node_labels) == len(valid_edge_index)
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

    state_dict = torch.load(config["reload_path"])
    gat.load_state_dict(state_dict["state_dict"])
    print("reload from ", config["reload_path"])
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

    # Step 4: Start the testing procedure

    node_label_2_id = json.load(open("node_label_dict.json"))
    id_2_node_label = {}
    for k, v in node_label_2_id.items():
        id_2_node_label[v] = k
    # Validation loop
    with torch.no_grad():
        try:
            print("test starting...")
            valid_graph_data = (valid_node_features, valid_node_labels, valid_edge_index)
            time_start = time.time()
            class_pd_list, loss, accuracy = main_loop(LoopPhase.VAL, valid_graph_data, epoch=0)
            print("GAT test: time elapsed=" + str(time.time() - time_start) + "[s]|test acc=" + str(accuracy))
            print("pic drawing")
            for i, diagram in enumerate(class_pd_list):
                idx2originid = json.load(open(os.path.join(config["id2originid_json_path"], valid_file_name[i])))
                all_trace = json.load(open(os.path.join(config["trace_path"], valid_file_name[i])))
                # print("diagram", diagram)
                # print("all_trace", all_trace)
                # print("idx2originid", idx2originid)
                plt.gca().invert_yaxis()
                label_already_dict = {}
                for j, node_pd in enumerate(diagram):
                    x, y = zip(*all_trace[idx2originid[str(j)]]["coords"])
                    if node_pd == node_label_2_id["connection"]:
                        color = "#054E9F"
                    elif node_pd == node_label_2_id["arrow"]:
                        color = "#F2A90D"
                    elif node_pd == node_label_2_id["data"]:
                        color = "#F20D43"
                    elif node_pd == node_label_2_id["text"]:
                        color = "#F20DC4"
                    elif node_pd == node_label_2_id["process"]:
                        color = "#9F0DF2"
                    elif node_pd == node_label_2_id["terminator"]:
                        color = "#0D23F2"
                    elif node_pd == node_label_2_id["decision"]:
                        color = "#0DDFF2"
                    else:
                        raise Exception('error node label type')
                    if not id_2_node_label[node_pd.item()] in label_already_dict:
                        plt.plot(x, y, linewidth=2, c=color, label=id_2_node_label[node_pd.item()])
                        label_already_dict[id_2_node_label[node_pd.item()]] = 1
                    else:
                        plt.plot(x, y, linewidth=2, c=color)
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(config["pic_path"], valid_file_name[i][:-5]) + ".jpg")
                plt.gcf().clear()

        except Exception as e:  # "patience has run out" exception :O
            traceback.print_exc()


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
    parser.add_argument("--reload_path", type=str,
                        help="the path of reload model")
    parser.add_argument("--trace_json_path", type=str,
                        help="the path of origin trace")
    parser.add_argument("--id2originid_json_path", type=str, help="the path of id2originid file")
    parser.add_argument("--trace_path", type=str, help="the path of origin trace file")
    parser.add_argument("--pic_path", type=str, help="the path of output pic")
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
    test_gat_cora(get_training_args())
