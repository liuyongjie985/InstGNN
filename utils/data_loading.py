"""
    Currently I only have support for Cora dataset - feel free to add your own graph data.
    You can find the details on how Cora was constructed here: http://eliassi.org/papers/ai-mag-tr08.pdf

    TL;DR:
    The feature vectors are 1433 features long. The authors found the most frequent words across every paper
    in the graph (they've removed the low frequency words + some additional processing) and made a vocab from those.
    Now feature "i" in the feature vector tells us whether the paper contains i-th word from the vocab (1-yes, 0-no).
    e.g. : feature vector 100...00 means that this node/paper contains only the 0th word of the vocab.


    Note on Cora processing:
        GAT and many other papers (GCN, etc.) used the processed version of Cora that can be found here:
        https://github.com/kimiyoung/planetoid

        I started from that same data, and after pre-processing it the same way as GAT and GCN,
        I've saved it into only 3 files so there is no need to copy-paste the same pre-processing code around anymore.

        Node features are saved in CSR sparse format, labels go from 0-6 (not one-hot) and finally the topology of the
        graph remained the same I just renamed it to adjacency_list.dict.


    Note on sparse matrices:
        If you're like me you didn't have to deal with sparse matrices until you started playing with GNNs.
        You'll usually see the following formats in GNN implementations: LIL, COO, CSR and CSC.
        Occasionally, you'll also see DOK and in special settings DIA and BSR as well (so 7 in total).

        It's not nuclear physics (it's harder :P) check out these 2 links and you're good to go:
            * https://docs.scipy.org/doc/scipy/reference/sparse.html
            * https://en.wikipedia.org/wiki/Sparse_matrix

        TL;DR:
        LIL, COO and DOK are used for efficient modification of your sparse structure/graph topology (add/remove edges)
        CSC and CSR are used for efficient arithmetic operations (addition, multiplication, etc.)
        DIA and BSR are used when you're dealing with special types of sparse matrices - diagonal and block matrices.

"""

import pickle
import zipfile
import json

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import torch
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset

from utils.constants import *
from utils.visualizations import plot_in_out_degree_distributions, visualize_graph


def load_graph_data(training_config, device):
    dataset_name = training_config['dataset_name'].lower()
    layer_type = training_config['layer_type']
    should_visualize = training_config['should_visualize']

    if dataset_name == DatasetType.CORA.name.lower():  # Cora citation network

        # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        # shape = (N, 1)
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
        # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
        adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))

        # Normalize the features
        node_features_csr = normalize_features_sparse(node_features_csr)
        num_of_nodes = len(node_labels_npy)

        if layer_type == LayerType.IMP3:
            # Build edge index explicitly (faster than nx ~100 times and as fast as PyGeometric imp but less complex)
            # shape = (2, E), where E is the number of edges, and 2 for source and target nodes. Basically edge index
            # contains tuples of the format S->T, e.g. 0->3 means that node with id 0 points to a node with id 3.
            topology = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True)
        elif layer_type == LayerType.IMP2 or layer_type == LayerType.IMP1:
            # adjacency matrix shape = (N, N)
            topology = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list_dict)).todense().astype(np.float)
            topology += np.identity(topology.shape[0])  # add self connections
            topology[topology > 0] = 1  # multiple edges not allowed
            topology[topology == 0] = -np.inf  # make it a mask instead of adjacency matrix (used to mask softmax)
            topology[topology == 1] = 0
        else:
            raise Exception(f'Layer type {layer_type} not yet supported.')

        # Note: topology is just a fancy way of naming the graph structure data
        # (be it in the edge index format or adjacency matrix)

        if should_visualize:  # network analysis and graph drawing
            plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)
            visualize_graph(topology, node_labels_npy, dataset_name)

        # Convert to dense PyTorch tensors

        # Needs to be long int type (in implementation 3) because later functions like PyTorch's index_select expect it
        topology = torch.tensor(topology, dtype=torch.long if layer_type == LayerType.IMP3 else torch.float,
                                device=device)
        node_labels = torch.tensor(node_labels_npy, dtype=torch.long, device=device)  # Cross entropy expects a long int
        node_features = torch.tensor(node_features_csr.todense(), device=device)

        # Indices that help us extract nodes that belong to the train/val and test splits
        train_indices = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
        val_indices = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
        test_indices = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)

        return node_features, node_labels, topology, train_indices, val_indices, test_indices

    elif dataset_name == DatasetType.PPI.name.lower():  # Protein-Protein Interaction dataset

        # Instead of checking PPI in, I'd rather download it on-the-fly the first time it's needed (lazy execution ^^)
        if not os.path.exists(PPI_PATH):  # download the first time this is ran
            os.makedirs(PPI_PATH)

            # Step 1: Download the ppi.zip (contains the PPI dataset)
            zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
            download_url_to_file(PPI_URL, zip_tmp_path)

            # Step 2: Unzip it
            with zipfile.ZipFile(zip_tmp_path) as zf:
                zf.extractall(path=PPI_PATH)
            print(f'Unzipping to: {PPI_PATH} finished.')

            # Step3: Remove the temporary resource file
            os.remove(zip_tmp_path)
            print(f'Removing tmp file {zip_tmp_path}.')

        # Collect train/val/test graphs here
        edge_index_list = []
        node_features_list = []
        node_labels_list = []

        # Dynamically determine how many graphs we have per split (avoid using constants when possible)
        num_graphs_per_split_cumulative = [0]

        # Small optimization "trick" since we only need test in the playground.py
        splits = ['test'] if training_config['ppi_load_test_only'] else ['train', 'valid', 'test']

        for split in splits:
            # PPI has 50 features per node, it's a combination of positional gene sets, motif gene sets,
            # and immunological signatures - you can treat it as a black box (I personally have a rough understanding)
            # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
            # Note: node features are already preprocessed
            node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

            # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
            # SHAPE = (NS, 121)
            node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

            # Graph topology stored in a special nodes-links NetworkX format
            nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
            # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
            # The reason I use a NetworkX's directed graph is because we need to explicitly model both directions
            # because of the edge index and the way GAT implementation #3 works
            collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
            # For each node in the above collection, ids specify to which graph the node belongs to
            graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
            num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

            # Split the collection of graphs into separate PPI graphs
            for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
                mask = graph_ids == graph_id  # find the nodes which belong to the current graph (identified via id)
                graph_node_ids = np.asarray(mask).nonzero()[0]
                graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
                print(f'Loading {split} graph {graph_id} to CPU. '
                      f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

                # shape = (2, E) - where E is the number of edges in the graph
                # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
                # is a scarcer resource than CPU's RAM and the whole PPI dataset can't fit during the training.
                edge_index = torch.tensor(list(graph.edges), dtype=torch.long).transpose(0, 1).contiguous()
                edge_index = edge_index - edge_index.min()  # bring the edges to [0, num_of_nodes] range
                edge_index_list.append(edge_index)
                # shape = (N, 50) - where N is the number of nodes in the graph
                node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
                # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
                node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))

                if should_visualize:
                    plot_in_out_degree_distributions(edge_index.numpy(), graph.number_of_nodes(), dataset_name)
                    visualize_graph(edge_index.numpy(), node_labels[mask], dataset_name)

        #
        # Prepare graph data loaders
        #

        # Optimization, do a shortcut in case we only need the test data loader
        if training_config['ppi_load_test_only']:
            data_loader_test = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                batch_size=training_config['batch_size'],
                shuffle=False
            )
            return data_loader_test
        else:

            data_loader_train = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                batch_size=training_config['batch_size'],
                shuffle=True
            )

            data_loader_val = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                batch_size=training_config['batch_size'],
                shuffle=False  # no need to shuffle the validation and test graphs
            )

            data_loader_test = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                batch_size=training_config['batch_size'],
                shuffle=False
            )

            return data_loader_train, data_loader_val, data_loader_test
    else:
        raise Exception(f'{dataset_name} not yet supported.')


def sogou_load_node_feature(node_feature_path, device):
    node_feature_list = []
    file_name_list = []
    # print(node_feature_path)
    for parent, dirnames, filenames in os.walk(node_feature_path, followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":
                file_path = os.path.join(parent, filename)
                temp_node_feature = json.load(open(file_path))
                temp_node_feature = torch.tensor(temp_node_feature, device=device)
                # print("temp_node_feature.shape", temp_node_feature.shape)
                node_feature_list.append(temp_node_feature)
                file_name_list.append(filename)
    return file_name_list, node_feature_list


def sogou_load_node_label(node_label_path, label_dict, device):
    node_label_list = []
    for parent, dirnames, filenames in os.walk(node_label_path, followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":
                file_path = os.path.join(parent, filename)
                # N * 2
                temp_node_label = json.load(open(file_path))
                temp_list = []
                for single_node_label in temp_node_label:
                    # if single_node_label[1] in label_dict:
                    #     temp_list.append(label_dict[single_node_label[1]])
                    # else:
                    #     label_dict[single_node_label[1]] = len(label_dict)
                    temp_list.append(label_dict[single_node_label[1]])
                temp_tensor = torch.tensor(temp_list, dtype=torch.long,
                                           device=device)  # Cross entropy expects a long int
                # print("temp_tensor.shape", temp_tensor.shape)
                node_label_list.append(temp_tensor)
    return node_label_list


'''
N * N，里面每个 item 不是 None 就是 19维特征
'''


def edgeTwoFeatureTrans2Matrix(edge_features, edge_label):
    feature_result = []
    connection_result = []
    sparse_feature_result = []
    sparse_feature_index_result = []
    sparse_label_result = []
    for i, x in enumerate(edge_features):
        temp_list = []
        temp_feature_list = []

        for j, y in enumerate(x):
            if y == None:
                temp_list.append(0)
                temp_feature_list.append([0 for x in range(21)])
            else:
                temp_list.append(1)
                temp_feature_list.append(y)
                sparse_feature_result.append(y)
                sparse_feature_index_result.append([i, j])
                sparse_label_result.append(edge_label[i][j][0])

        feature_result.append(temp_feature_list)
        connection_result.append(temp_list)

    feature_result = np.array(feature_result, dtype=np.float32)
    connection_result = np.array(connection_result, dtype=np.float32)
    sparse_feature_result = np.array(sparse_feature_result, dtype=np.float32)
    sparse_feature_index_result = np.array(sparse_feature_index_result, dtype=np.int64)
    sparse_label_result = np.array(sparse_label_result, dtype=np.int64)

    connection_result[connection_result > 0] = 1  # multiple edges not allowed
    connection_result[
        connection_result == 0] = -np.inf  # make it a mask instead of adjacency matrix (used to mask softmax)
    connection_result[connection_result == 1] = 0
    return feature_result, connection_result, sparse_feature_result, sparse_feature_index_result, sparse_label_result


def edgeFeatureTrans2Matrix(edge_features):
    feature_result = []
    connection_result = []
    for x in edge_features:
        temp_list = []
        temp_feature_list = []
        for y in x:
            if y == None:
                temp_list.append(0)
                temp_feature_list.append([0 for x in range(21)])
            else:
                temp_list.append(1)
                temp_feature_list.append(y)
        feature_result.append(temp_feature_list)
        connection_result.append(temp_list)
    feature_result = np.array(feature_result, dtype=np.float32)
    connection_result = np.array(connection_result, dtype=np.float32)
    connection_result[connection_result > 0] = 1  # multiple edges not allowed
    connection_result[
        connection_result == 0] = -np.inf  # make it a mask instead of adjacency matrix (used to mask softmax)
    connection_result[connection_result == 1] = 0
    return feature_result, connection_result


'''
N * N 每个item是个tuple，tuple的第一维为0/1代表是否为同一个symbol，tuple第二维为symbol
'''


def edgeLabelTrans2Matrix(edge_labels):
    label_result = []
    for x in edge_labels:
        temp_list = []
        for y in x:
            if y[0] == 1:
                temp_list.append(1)
            else:
                temp_list.append(0)
        label_result.append(temp_list)
    label_result = np.array(label_result, dtype=np.float32)
    return label_result


def sparseEdgeLabelTrans2Matrix(edge_labels):
    label_result = []
    for x in edge_labels:
        temp_list = []
        for y in x:
            if y[0] == 1:
                temp_list.append(1)
        label_result.append(temp_list)
    label_result = np.array(label_result, dtype=np.float32)
    return label_result


def sogou_load_edge_features(edge_feature_path, device):
    edge_feature_list = []
    connection_list = []
    for parent, dirnames, filenames in os.walk(edge_feature_path, followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":
                file_path = os.path.join(parent, filename)
                temp_edge_features = json.load(open(file_path))
                temp_edge_features, connections = edgeFeatureTrans2Matrix(temp_edge_features)

                # Convert to dense PyTorch tensors
                # Needs to be long int type (in implementation 3) because later functions like PyTorch's index_select expect it
                temp_edge_features = torch.tensor(temp_edge_features,
                                                  dtype=torch.float,
                                                  device=device)

                connections = torch.tensor(connections,
                                           dtype=torch.float,
                                           device=device)

                edge_feature_list.append(temp_edge_features)
                connection_list.append(connections)

    return edge_feature_list, connection_list


def sogou_load_edge_features_and_labels(edge_feature_path, edge_label_path, device):
    edge_feature_list = []
    connection_list = []

    sparse_edge_feature_list = []
    sparse_edge_index_list = []
    sparse_edge_label_list = []
    for parent, dirnames, filenames in os.walk(edge_feature_path, followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":
                edge_feature_file_path = os.path.join(parent, filename)
                temp_edge_features = json.load(open(edge_feature_file_path))
                edge_label_file_path = os.path.join(edge_label_path, filename)

                temp_edge_label = json.load(open(edge_label_file_path))
                temp_edge_features, connection_result, sparse_feature_result, sparse_feature_index_result, sparse_label_result = edgeTwoFeatureTrans2Matrix(
                    temp_edge_features, temp_edge_label)

                temp_edge_features = torch.tensor(temp_edge_features,
                                                  dtype=torch.float,
                                                  device=device)

                connections = torch.tensor(connection_result,
                                           dtype=torch.float,
                                           device=device)

                sparse_feature_result = torch.tensor(sparse_feature_result, dtype=torch.float, device=device)
                sparse_feature_index_result = torch.tensor(sparse_feature_index_result, dtype=torch.long, device=device)
                sparse_label_result = torch.tensor(sparse_label_result, dtype=torch.long, device=device)

                edge_feature_list.append(temp_edge_features)
                connection_list.append(connections)

                sparse_edge_feature_list.append(sparse_feature_result)
                sparse_edge_index_list.append(sparse_feature_index_result)
                sparse_edge_label_list.append(sparse_label_result)

    return edge_feature_list, connection_list, sparse_edge_feature_list, sparse_edge_index_list, sparse_edge_label_list


def sogou_load_edge_label(edge_label_path, device):
    edge_label_list = []
    for parent, dirnames, filenames in os.walk(edge_label_path, followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":
                file_path = os.path.join(parent, filename)
                temp_edge_label = json.load(open(file_path))
                temp_edge_label = edgeLabelTrans2Matrix(temp_edge_label)

                # Convert to dense PyTorch tensors
                # Needs to be long int type (in implementation 3) because later functions like PyTorch's index_select expect it
                temp_edge_label = torch.tensor(temp_edge_label,
                                               dtype=torch.long,
                                               device=device)
                edge_label_list.append(temp_edge_label)
    return edge_label_list


def load_sogou_sparse_graph_data(config, type, device):
    dataset_name = config['dataset_name'].lower()
    layer_type = config['layer_type']
    should_visualize = config['should_visualize']
    print("Loading " + type + " ", DatasetType.SOGOU.name.lower())
    if dataset_name == DatasetType.SOGOU.name.lower():  # Cora citation network
        # shape = (B, N, FIN), where N is the number of nodes and FIN is the number of input features
        file_name_list, node_features = sogou_load_node_feature(os.path.join(SOGOU_PATH, type + '/node_feature_json'),
                                                                device)
        node_label_dict = json.load(open("node_label_dict.json"))
        # shape = (B, N, 1)
        node_labels = sogou_load_node_label(os.path.join(SOGOU_PATH, type + '/node_label_json'), node_label_dict,
                                            device)
        # shape = (B, N, N)
        edge_features, edge_connections, sparse_edge_features, sparse_edge_indexs, sparse_edge_labels = sogou_load_edge_features_and_labels(
            os.path.join(SOGOU_PATH, type + '/edge_feature_json'), os.path.join(SOGOU_PATH, type + '/edge_label_json'),
            device)

        return file_name_list, node_features, node_labels, edge_features, edge_connections, sparse_edge_features, sparse_edge_indexs, sparse_edge_labels
    else:
        raise Exception(f'{dataset_name} not yet supported.')


def load_sogou_graph_data(config, type, device):
    dataset_name = config['dataset_name'].lower()
    layer_type = config['layer_type']
    should_visualize = config['should_visualize']
    print("Loading " + type + " ", DatasetType.SOGOU.name.lower())
    if dataset_name == DatasetType.SOGOU.name.lower():  # Cora citation network
        # shape = (B, N, FIN), where N is the number of nodes and FIN is the number of input features
        file_name_list, node_features = sogou_load_node_feature(os.path.join(SOGOU_PATH, type + '/node_feature_json'),
                                                                device)
        node_label_dict = json.load(open("node_label_dict.json"))
        # shape = (B, N, 1)
        node_labels = sogou_load_node_label(os.path.join(SOGOU_PATH, type + '/node_label_json'), node_label_dict,
                                            device)
        # json.dump(node_label_dict, open("node_label_dict.json", "w"))
        # shape = (B, N, N)
        edge_features, edge_connection = sogou_load_edge_features(os.path.join(SOGOU_PATH, type + '/edge_feature_json'),
                                                                  device)
        edge_label = sogou_load_edge_label(os.path.join(SOGOU_PATH, type + '/edge_label_json'), device)
        # Note: topology is just a fancy way of naming the graph structure data
        # (be it in the edge index format or adjacency matrix)
        if should_visualize:  # network analysis and graph drawing
            plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)
            visualize_graph(topology, node_labels_npy, dataset_name)
        # print("node_labels", node_labels)
        return file_name_list, node_features, node_labels, edge_features, edge_connection, edge_label
    else:
        raise Exception(f'{dataset_name} not yet supported.')


class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """

    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """

    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_features, node_labels, edge_index


def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data


# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


'''
每行向量按照自己的词语数量做平均
'''


def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

    # Instead of dividing (like in normalize_features_dense()) we do multiplication with inverse sum of features.
    # Modern hardware (GPUs, TPUs, ASICs) is optimized for fast matrix multiplications! ^^ (* >> /)
    # shape = (N, FIN) -> (N, 1), where N number of nodes and FIN number of input features
    node_features_sum = np.array(node_features_sparse.sum(-1))  # sum features for every node feature vector

    # Make an inverse (remember * by 1/x is better (faster) then / by x)
    # shape = (N, 1) -> (N)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # Again certain sums will be 0 so 1/0 will give us inf so we replace those by 1 which is a neutral element for mul
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.

    # Create a diagonal matrix whose values on the diagonal come from node_features_inv_sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    # We return the normalized features.
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


# Not used -> check out playground.py where it is used in profiling functions
def normalize_features_dense(node_features_dense):
    assert isinstance(node_features_dense, np.matrix), f'Expected np matrix got {type(node_features_dense)}.'

    # The goal is to make feature vectors normalized (sum equals 1), but since some feature vectors are all 0s
    # in those cases we'd have division by 0 so I set the min value (via np.clip) to 1.
    # Note: 1 is a neutral element for division i.e. it won't modify the feature vector
    return node_features_dense / np.clip(node_features_dense.sum(1), a_min=1, a_max=None)


def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)
                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index


# Not used - this is yet another way to construct the edge index by leveraging the existing package (networkx)
# (it's just slower than my simple implementation build_edge_index())
def build_edge_index_nx(adjacency_list_dict):
    nx_graph = nx.from_dict_of_lists(adjacency_list_dict)
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj.tocoo()  # convert to COO (COOrdinate sparse format)
    return np.row_stack((adj.row, adj.col))
