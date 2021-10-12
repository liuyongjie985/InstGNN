import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from models.definitions.InstGNN_4 import InstGNN
from utils.data_loading import load_sogou_sparse_graph_data
from utils.constants import *
from utils.utils import get_instgnn4_main_loop
import json
import matplotlib.pyplot as plt
import traceback
import numpy as np


def test_gat_cora(config):
    global BEST_VAL_PERF, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    # node_feature B N 27
    # node_labels B N
    # edge_index B N * N

    valid_file_name, valid_node_features, valid_node_labels, valid_edge_features, valid_edge_connections, valid_sparse_edge_features, valid_sparse_edge_indexs, valid_sparse_edge_labels = load_sogou_sparse_graph_data(
        config, "valid", device)
    assert len(valid_node_features) == len(valid_node_labels) == len(valid_edge_features) == len(
        valid_edge_connections)

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

    state_dict = torch.load(config["reload_path"])
    instgnn.load_state_dict(state_dict["state_dict"])
    print("reload from ", config["reload_path"])
    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(instgnn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_instgnn4_main_loop(
        config,
        instgnn,
        loss_fn, None,
        optimizer, device)

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
            valid_graph_data = (
                valid_node_features, valid_node_labels, valid_edge_features, valid_edge_connections,
                valid_sparse_edge_features, valid_sparse_edge_indexs, valid_sparse_edge_labels)
            time_start = time.time()

            node_class_pd_list, edge_class_pd_list, loss, node_accuracy, edge_accuracy = main_loop(LoopPhase.VAL,
                                                                                                   valid_graph_data,
                                                                                                   epoch=0)
            print("GAT test: time elapsed=" + str(time.time() - time_start) + "[s]|test node acc=" + str(
                node_accuracy) + "|test edge acc=" + str(edge_accuracy))
            print("pic drawing")
            for i, diagram in enumerate(node_class_pd_list):
                # for i, diagram in enumerate(valid_node_labels):
                idx2originid = json.load(open(os.path.join(config["id2originid_json_path"], valid_file_name[i])))
                all_trace = json.load(open(os.path.join(config["trace_path"], valid_file_name[i])))
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
                plt.savefig(os.path.join(config["node_pic_path"], valid_file_name[i][:-5]) + ".jpg")
                plt.gcf().clear()

                # =========== old edge classification =========
                # plt.gca().invert_yaxis()
                # group_list = []
                # already_group = {}
                # for j, i_edges in enumerate(edge_class_pd_list[i]):
                # for j, i_edges in enumerate(valid_edge_label[i]):
                #     if j not in already_group:
                #         temp_group = [j]
                #         already_group[j] = 1
                #         for k, _ in enumerate(i_edges):
                #             temp_item = _.item()
                #             if temp_item == 1:
                #                 temp_group.append(k)
                #                 already_group[k] = 1
                #         group_list.append(temp_group)
                #
                # for temp_group in group_list:
                #     color = np.random.rand(3, )
                #     for stroke_id in temp_group:
                #         x, y = zip(*all_trace[idx2originid[str(stroke_id)]]["coords"])
                #         plt.plot(x, y, linewidth=2, c=color)
                # plt.savefig(os.path.join(config["edge_pic_path"], valid_file_name[i][:-5]) + ".jpg")
                # plt.gcf().clear()
                # =========== new edge classification =========

                edge_matrix = np.zeros(shape=valid_edge_connections[i].shape)
                assert len(edge_class_pd_list[i]) == len(valid_sparse_edge_indexs[i])
                group_list = []
                already_dict = {}
                abc = 0
                # for j, i_sparse_edges in enumerate(valid_sparse_edge_labels[i]):
                for j, i_sparse_edges in enumerate(edge_class_pd_list[i]):
                    if i_sparse_edges == 1:
                        temp_x = valid_sparse_edge_indexs[i][j][0].item()
                        temp_y = valid_sparse_edge_indexs[i][j][1].item()
                        print("x y", valid_sparse_edge_indexs[i][j])
                        print("already_dict", already_dict)

                        if temp_x in already_dict and temp_y in already_dict:
                            if already_dict[temp_x] != already_dict[temp_y]:
                                group_list[already_dict[temp_x]].extend(group_list[already_dict[temp_y]])
                                for group_item in group_list[already_dict[temp_y]]:
                                    already_dict[group_item] = already_dict[temp_x]
                        else:
                            if temp_x in already_dict:
                                group_list[already_dict[temp_x]].append(temp_y)
                                already_dict[temp_y] = already_dict[temp_x]
                            if temp_y in already_dict:
                                group_list[already_dict[temp_y]].append(temp_x)
                                already_dict[temp_x] = already_dict[temp_y]
                            if temp_x not in already_dict and temp_y not in already_dict:
                                already_dict[temp_x] = len(group_list)
                                already_dict[temp_y] = len(group_list)
                                group_list.append([temp_x, temp_y])
                        print("group_list", group_list)
                        print(abc)
                        abc += 1
                        # if abc >= 10:
                        #     exit()
                plt.gca().invert_yaxis()
                pic_already_dict = {}
                print("group_list", group_list)
                print("already_dict", already_dict)
                for temp_group in group_list:
                    if already_dict[temp_group[0]] not in pic_already_dict:
                        color = np.random.rand(3, )
                        for stroke_id in temp_group:
                            x, y = zip(*all_trace[idx2originid[str(stroke_id)]]["coords"])
                            plt.plot(x, y, linewidth=2, c=color)
                        pic_already_dict[already_dict[temp_group[0]]] = 1

                plt.savefig(os.path.join(config["edge_pic_path"], valid_file_name[i][:-5]) + ".jpg")
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
    parser.add_argument("--node_pic_path", type=str, help="the path of output node pic")
    parser.add_argument("--edge_pic_path", type=str, help="the path of output edge pic")
    args = parser.parse_args()

    # Model architecture related
    # INSTGNN
    instgnn_config = {
        "num_of_layers": 3,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_joint_learning_layers": 1,
        "num_heads_per_layer": [8, 8, 8],
        "num_features_per_layer": [SOGOU_NUM_INPUT_FEATURES, 32, 32, 32],
        "jll_num_heads_per_layer": [8, 1],
        "jll_num_features_per_layer": [32, SOGOU_NUM_CLASSES],
        "num_edge_features_per_layer": SOGOU_NUM_INPUT_EDGE_FEATURES,
        "jll_edge_num_features": [SOGOU_NUM_INPUT_EDGE_FEATURES, SOGOU_EDGE_NUM_CLASS],
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
    training_config.update(instgnn_config)
    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    test_gat_cora(get_training_args())
