import re
import numpy as np
import torch
from utils.constants import *
from shapely.geometry import Point, Polygon


def isStrokeInGroup(s, gxs, gys):
    coords = [(gxs[i], gys[i]) for i in range(len(gxs))]
    poly = Polygon(coords)
    in_num = 0
    total_num = 0

    for point in s:
        p = Point(point["x"], point["y"])
        if p.within(poly):
            in_num += 1
        total_num += 1

    if total_num != 0 and in_num / total_num > 0.8:
        return True
    else:
        return False


def convert_adj_to_edge_index(adjacency_matrix):
    """
    Handles both adjacency matrices as well as connectivity masks used in softmax (check out Imp2 of the GAT model)
    Connectivity masks are equivalent to adjacency matrices they just have -inf instead of 0 and 0 instead of 1.
    I'm assuming non-weighted (binary) adjacency matrices here obviously and this code isn't meant to be as generic
    as possible but a learning resource.

    """
    assert isinstance(adjacency_matrix, np.ndarray), f'Expected NumPy array got {type(adjacency_matrix)}.'
    height, width = adjacency_matrix.shape
    assert height == width, f'Expected square shape got = {adjacency_matrix.shape}.'

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] == active_value:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)


def name_to_layer_type(name):
    if name == LayerType.IMP1.name:
        return LayerType.IMP1
    elif name == LayerType.IMP2.name:
        return LayerType.IMP2
    elif name == LayerType.IMP3.name:
        return LayerType.IMP3
    else:
        raise Exception(f'Name {name} not supported.')


def get_training_state(training_config, model):
    training_state = {
        "commit_hash": "",
        # Training details
        "dataset_name": training_config['dataset_name'],
        "num_of_epochs": training_config['num_of_epochs'],
        "test_perf": training_config['test_perf'],

        # Model structure
        "num_of_layers": training_config['num_of_layers'],
        "num_heads_per_layer": training_config['num_heads_per_layer'],
        "num_features_per_layer": training_config['num_features_per_layer'],
        "add_skip_connection": training_config['add_skip_connection'],
        "bias": training_config['bias'],
        "dropout": training_config['dropout'],
        "layer_type": training_config['layer_type'].name,

        # Model state
        "state_dict": model.state_dict()
    }

    return training_state


def get_available_binary_name(dataset_name='unknown'):
    prefix = f'gat_{dataset_name}'

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def print_model_metadata(training_state):
    header = f'\n{"*" * 5} Model training metadata: {"*" * 5}'
    print(header)

    for key, value in training_state.items():
        if key != 'state_dict':  # don't print state_dict it's a bunch of numbers...
            print(f'{key}: {value}')
    print(f'{"*" * len(header)}\n')


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_instgnn4_main_loop(config, instgnn, cross_entropy_loss, mse_loss, optimizer, device):
    def main_loop(phase, graph_data, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer
        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            instgnn.train()
        else:
            instgnn.eval()

        NUM_BATCH = len(graph_data[0])
        total_loss = 0
        total_node_accuracy = 0
        total_edge_accuracy = 0
        node_class_pd_list = []
        edge_class_pd_list = []
        for single_graph_data in zip(*graph_data):
            # train_graph_data = (train_node_features, train_node_labels, train_edge_features, train_edge_connections,train_sparse_edge_features, train_sparse_edge_indexs, train_sparse_edge_labels)
            out_nodes_features, in_edges_features, connectivity_mask, out_edges_features, sparse_edges_indexs = instgnn(
                (single_graph_data[0], single_graph_data[2], single_graph_data[3], single_graph_data[4],
                 single_graph_data[5]))

            # N*1
            gt_node_labels = single_graph_data[1]  # gt stands for ground truth
            gt_edge_labels = single_graph_data[6]
            # Example: let's take an output for a single node on Cora - it's a vector of size 7 and it contains unnormalized
            # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
            # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
            # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
            # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
            # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
            # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
            cross_entropy_loss.weight = None
            loss_nc = cross_entropy_loss(out_nodes_features, gt_node_labels)

            # Undirected graph require diagonal matrix
            edge_class_predictions = torch.argmax(out_edges_features, dim=-1)
            edge_class_pd_list.append(edge_class_predictions)

            weight1 = gt_edge_labels.clone()
            weight1 = torch.sum(weight1)
            # print("gt_edge_labels", gt_edge_labels)
            weight0 = len(gt_edge_labels.clone()) - weight1

            weight0 = 1 / (weight0) if weight0 != 0 else torch.tensor(0, device=device)
            weight1 = 1 / (weight1) if weight1 != 0 else torch.tensor(0, device=device)

            cross_entropy_loss.weight = torch.tensor(np.array([weight0.cpu(), weight1.cpu()]), device=device,
                                                     dtype=torch.float32)
            loss_ec = cross_entropy_loss(out_edges_features, gt_edge_labels)

            loss = loss_ec + loss_nc
            total_loss += loss.item()
            if phase == LoopPhase.TRAIN:
                optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                optimizer.step()  # apply the gradients to weights

            # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
            # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
            node_class_predictions = torch.argmax(out_nodes_features, dim=-1)
            node_class_pd_list.append(node_class_predictions)
            node_accuracy = torch.sum(torch.eq(node_class_predictions, gt_node_labels).long()).item() / len(
                gt_node_labels)
            # E
            edge_accuracy = torch.sum(torch.eq(edge_class_predictions, gt_edge_labels).long()).item() / len(
                gt_edge_labels)

            total_node_accuracy += node_accuracy
            total_edge_accuracy += edge_accuracy
            # Logging
            if phase == LoopPhase.TRAIN:
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), epoch)
                    writer.add_scalar('training_node_acc', node_accuracy, epoch)
                    writer.add_scalar('training_edge_acc', edge_accuracy, epoch)


            elif phase == LoopPhase.VAL:
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), epoch)
                    writer.add_scalar('val_node_acc', node_accuracy, epoch)
                    writer.add_scalar('val_edge_acc', edge_accuracy, epoch)

        return node_class_pd_list, edge_class_pd_list, total_loss / NUM_BATCH, total_node_accuracy / NUM_BATCH, total_edge_accuracy / NUM_BATCH

    return main_loop  # return the decorated function
