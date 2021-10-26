import torch
import torch.nn as nn

from utils.constants import LayerType


class InstGNN(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.

    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.

    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.
    """

    def __init__(self, num_of_layers, num_of_joint_learning_layers, jll_num_heads_per_layer, jll_num_features_per_layer,
                 num_heads_per_layer, num_features_per_layer,
                 num_edge_features_per_layer, jll_edge_num_features,
                 add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'
        assert num_of_joint_learning_layers == len(jll_num_heads_per_layer) - 1 == len(
            jll_num_features_per_layer) - 1, f'Enter valid arch params.'
        InstGNNLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        instgnn_backbone_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = InstGNNLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_in_edge_features=num_edge_features_per_layer,
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU(),  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            instgnn_backbone_layers.append(layer)

        self.backbone_net = nn.Sequential(
            *instgnn_backbone_layers,
        )
        joint_learning_layers = []
        for i in range(num_of_joint_learning_layers):
            temp_joint_learning_layer = JointLearningLayer(
                ell_edge_num_in_features=jll_edge_num_features[i],
                # ell_sparse_edge_num_in_features=jll_edge_num_features[i],
                ell_edge_num_out_features=jll_edge_num_features[i + 1],
                ell_node_num_in_features=jll_num_features_per_layer[i + 1] * jll_num_heads_per_layer[i + 1],
                nll_node_num_in_features=jll_num_features_per_layer[i] * jll_num_heads_per_layer[i],
                nll_node_num_out_features=jll_num_features_per_layer[i + 1],
                nll_node_num_of_heads=jll_num_heads_per_layer[i + 1],
                layer_type=LayerType.IMP2,
                dropout_prob=dropout,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                log_attention_weights=log_attention_weights,
                concat=True if i < num_of_layers - 1 else False,
                add_skip_connection=add_skip_connection,
                bias=bias,
            )
            joint_learning_layers.append(temp_joint_learning_layer)
        self.joint_learning_net = nn.Sequential(*joint_learning_layers)

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        # out_nodes_features, in_edges_features, connectivity_mask
        backbone_out = self.backbone_net(data)
        jll_out = self.joint_learning_net(backbone_out)
        return jll_out


class JointLearningLayer(torch.nn.Module):
    head_dim = 1

    # num_in_edge_features 始终不变
    def __init__(self, ell_edge_num_in_features, ell_edge_num_out_features, ell_node_num_in_features,
                 nll_node_num_in_features, nll_node_num_out_features, nll_node_num_of_heads, layer_type, concat=True,
                 activation=nn.ELU(), dropout_prob=0.6, add_skip_connection=True, bias=True,
                 log_attention_weights=False):

        super().__init__()
        self.nll_node_num_of_heads = nll_node_num_of_heads
        self.nll_node_num_out_features = nll_node_num_out_features
        self.nll_node_num_in_features = nll_node_num_in_features

        self.ell_edge_num_in_features = ell_edge_num_in_features
        self.ell_edge_num_out_features = ell_edge_num_out_features
        self.ell_node_num_in_features = ell_node_num_in_features

        self.concat = concat
        self.add_skip_connection = add_skip_connection

        self.W_f1 = nn.Linear(ell_edge_num_in_features, ell_edge_num_out_features)
        self.W_f2 = nn.Linear(nll_node_num_out_features * nll_node_num_of_heads, ell_edge_num_out_features, bias=False)
        self.W_f3 = nn.Linear(ell_edge_num_out_features * 2, ell_edge_num_out_features, bias=False)
        self.W_f4 = nn.Linear(ell_edge_num_in_features, ell_edge_num_out_features)

        self.linear_proj = nn.Linear(nll_node_num_in_features, nll_node_num_of_heads * nll_node_num_out_features,
                                     bias=False)

        self.linear_proj2 = nn.Linear(ell_edge_num_in_features, nll_node_num_of_heads * nll_node_num_out_features)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, nll_node_num_of_heads, nll_node_num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, nll_node_num_of_heads, nll_node_num_out_features))
        self.scoring_edge_fn_target = nn.Parameter(torch.Tensor(1, 1, nll_node_num_of_heads, nll_node_num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(nll_node_num_of_heads * nll_node_num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(nll_node_num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(nll_node_num_in_features, nll_node_num_of_heads * nll_node_num_out_features,
                                       bias=False)

            self.edge_skip_proj = nn.Linear(ell_edge_num_in_features, ell_edge_num_out_features, bias=False)

        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        self.norm_node = nn.BatchNorm1d(nll_node_num_of_heads * nll_node_num_out_features, track_running_stats=False) \
            if concat else nn.BatchNorm1d(nll_node_num_out_features, track_running_stats=False)
        self.norm_edge = nn.BatchNorm1d(ell_edge_num_out_features, track_running_stats=False)
        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj2.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_edge_fn_target)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, data):
        # ============================================node learning layer===========================================
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        # data: (node_features, edge_index)
        in_nodes_features, in_edges_features, connectivity_mask, sparse_edge_features, sparse_edge_indexs = data  # unpack data

        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # in_nodes_features shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        dropout_nodes_features = self.dropout(in_nodes_features)
        # N * N * FIN
        dropout_edges_features = self.dropout(in_edges_features)

        nodes_features_proj = self.linear_proj(dropout_nodes_features).view(-1, self.nll_node_num_of_heads,
                                                                            self.nll_node_num_out_features)
        edges_features_proj = self.linear_proj2(dropout_edges_features).view(
            num_of_nodes, num_of_nodes, self.nll_node_num_of_heads,
            self.nll_node_num_out_features)

        edges_features_proj = self.leakyReLU(edges_features_proj)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        edges_features_proj = self.dropout(edges_features_proj)

        #
        # Step 2: Edge attention calculation (using sum instead of bmm + additional permute calls - compared to imp1)
        #
        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # nodes_features_proj shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        # edges_features_proj shape = (N, N, NH, FOUT) * (1,1,NH, FOUT) -> (N, N, NH)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        scores_edge_target = torch.sum((edges_features_proj * self.scoring_edge_fn_target), dim=-1)
        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)
        scores_edge_target = scores_edge_target.permute(2, 0, 1)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)
        all_edge_scores = self.leakyReLU(scores_edge_target)
        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + all_edge_scores + connectivity_mask)
        # Step 3: Neighborhood aggregation (same as in imp1)

        # batch matrix multiply, shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        # Note: watch out here I made a silly mistake of using reshape instead of permute thinking it will
        # end up doing the same thing, but it didn't! The acc on Cora didn't go above 52%! (compared to reported ~82%)
        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        #
        # Step 4: Residual/skip connections, concat and bias (same as in imp1)
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        # print("out_nodes_features.shape", out_nodes_features.shape)
        # ============================================edge learning layer===========================================

        dropout_nodes_features = self.dropout(out_nodes_features)
        # N * 1 * FIN
        node_source_feature = dropout_nodes_features.unsqueeze(1)
        # 1 * N * FIN
        node_target_feature = dropout_nodes_features.unsqueeze(0)

        dropout_sparse_edges_features = self.dropout(sparse_edge_features)
        # E << N * N
        # E * 21 => E * FOUT
        m_ij = self.W_f1(dropout_sparse_edges_features)
        # E * FOUT
        m_ij = self.leakyReLU(m_ij)
        # N * N * FOUT
        r_ij = self.W_f2(
            (node_source_feature - node_target_feature) * (node_source_feature - node_target_feature))
        # N * N * FOUT
        r_ij = self.leakyReLU(r_ij)
        # sparse_edge_index : E * 2 r_ij: E * FOUT
        r_ij = r_ij[sparse_edge_indexs.select(dim=1, index=0), sparse_edge_indexs.select(dim=1, index=1)]
        # E * FOUT
        out_edges_features = self.W_f3(torch.cat([m_ij, r_ij], dim=-1))
        if self.add_skip_connection:
            out_edges_features += self.edge_skip_proj(sparse_edge_features)

        if not (self.activation is None):
            out_edges_features = self.norm_edge(out_edges_features)

        out_edges_features = out_edges_features if self.activation is None else self.activation(out_edges_features)
        # N * N * FIN
        in_edges_features = self.W_f4(in_edges_features) * self.softmax(connectivity_mask).unsqueeze(-1)

        in_edges_features[
            sparse_edge_indexs.select(dim=1, index=0), sparse_edge_indexs.select(dim=1, index=1)] = out_edges_features
        # =====================================================================================================
        return out_nodes_features, in_edges_features, connectivity_mask, out_edges_features, sparse_edge_indexs

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.nll_node_num_of_heads,
                                                                             self.nll_node_num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1,
                                                         self.nll_node_num_of_heads * self.nll_node_num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if not (self.activation is None):
            out_nodes_features = self.norm_node(out_nodes_features)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class InstGNNLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.

    """
    head_dim = 1

    def __init__(self, num_in_features, num_in_edge_features, num_out_features, num_of_heads, layer_type, concat=True,
                 activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)

        if layer_type == LayerType.IMP1:
            # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            # You can treat this one matrix as num_of_heads independent W matrices
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
            self.linear_proj2 = nn.Linear(num_in_edge_features, num_of_heads * num_out_features)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_edge_fn_target = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))
            self.scoring_edge_fn_target = nn.Parameter(
                self.scoring_edge_fn_target.reshape(num_of_heads, num_out_features, 1, 1))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        self.norm = nn.BatchNorm1d(num_of_heads * num_out_features,
                                   track_running_stats=False) if concat else nn.BatchNorm1d(num_out_features,
                                                                                            track_running_stats=False)
        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj2.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_edge_fn_target)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    # attention_coefficients NH*N*N in_nodes_features N*DIM out_nodes_features N*NH*DOUT
    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)
        out_nodes_features = self.norm(out_nodes_features)
        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class InstGNNLayerImp2(InstGNNLayer):
    """
        Implementation #2 was inspired by the official GAT implementation: https://github.com/PetarV-/GAT

        It's conceptually simpler than implementation #3 but computationally much less efficient.

        Note: this is the naive implementation not the sparse one and it's only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.
    """

    def __init__(self, num_in_features, num_in_edge_features, num_out_features, num_of_heads, concat=True,
                 activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__(num_in_features, num_in_edge_features, num_out_features, num_of_heads, LayerType.IMP2, concat,
                         activation,
                         dropout_prob, add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        in_nodes_features, in_edges_features, connectivity_mask, edge_features, sparse_edge_indexs = data  # unpack data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # in_nodes_features shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)
        dropout_edges_features = self.dropout(in_edges_features)
        # nodes_features_proj shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # in_edge_features shape = (N, N, DIM) * (DIM, NH*FOUT) -> (N, N, NH, FOUT)
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        edges_features_proj = self.linear_proj2(
            dropout_edges_features.reshape(-1, dropout_edges_features.shape[2])).view(
            num_of_nodes, num_of_nodes, self.num_of_heads,
            self.num_out_features)

        edges_features_proj = self.leakyReLU(edges_features_proj)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        edges_features_proj = self.dropout(edges_features_proj)

        #
        # Step 2: Edge attention calculation (using sum instead of bmm + additional permute calls - compared to imp1)
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # nodes_features_proj shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        # edges_features_proj shape = (N, N, NH, FOUT) * (1,1,NH, FOUT) -> (N, N, NH)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        scores_edge_target = torch.sum((edges_features_proj * self.scoring_edge_fn_target), dim=-1)

        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)
        scores_edge_target = scores_edge_target.permute(2, 0, 1)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)
        all_edge_scores = self.leakyReLU(scores_edge_target)
        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + all_edge_scores + connectivity_mask)

        # Step 3: Neighborhood aggregation (same as in imp1)
        #

        # batch matrix multiply, shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        # Note: watch out here I made a silly mistake of using reshape instead of permute thinking it will
        # end up doing the same thing, but it didn't! The acc on Cora didn't go above 52%! (compared to reported ~82%)
        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        #
        # Step 4: Residual/skip connections, concat and bias (same as in imp1)
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, in_edges_features, connectivity_mask, edge_features, sparse_edge_indexs)


#
# Helper functions
#
def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    # if layer_type == LayerType.IMP1:
    #     return GATLayerImp1
    if layer_type == LayerType.IMP2:
        return InstGNNLayerImp2
    # elif layer_type == LayerType.IMP3:
    #     return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')
