"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.out_size = out_size
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z_list):
        w = torch.concat([self.project(z).mean(0) for z in z_list])     # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)

        z_final = torch.zeros(z_list[0].shape[0], z_list[0].shape[1])
        for i, z in enumerate(z_list):
            z_final += beta[i] * z
        return z_final

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads, out_size=out_size)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        # # FOR TESTING
        # for meta_path in self.meta_paths:
        #     print(meta_path)
        #     print(self._cached_coalesced_graph[meta_path].num_nodes)
        #     print(h[meta_path[0][0]].shape)
        #     print("\n\n")
        # exit(1)

        # Step 1: node-level attention via GAT
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            end_node = meta_path[0][0]
            # concatenation of results from all attention heads
            semantic_embeddings.append(self.gat_layers[i](
                new_g, h[end_node]).flatten(1))
            # SIMPLIFICATION: only compute attention for pairs
            #   (new_g x new_g), instead of (g x new_g)
            # this is due to limitations with the architecture

        # Step 2: semantic-level attention
        # SIMPLICATION: will only return a subset of node embeddings
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_sizes, proj_size, hidden_size, out_size, num_heads, dropout):
        """
        meta_paths: list of metapaths
        in_sizes: dictionary mapping node types to dimensionality
        proj_size: dimensionality of all nodes after projection
        hidden_size: dimensionality of semantic-specific node embedding
        out_size: dimensionality of output, typically number of classes
        num_heads: number of attn heads for node attention GATs
        dropout: dropout for node attention GATs
        """
        
        super(HAN, self).__init__()

        # Projection matrices
        self.proj = {}
        for ntype, dim in in_sizes.items():
            self.proj[ntype] = nn.Linear(dim, proj_size)

        # HAN Layers: node-attention and semantic-attention
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, proj_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        
        # MLP
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        """
        g: dgl graph.
        h: dict mapping node types to feature matrices
        """
        # Step 1: pass through projection layers
        proj_out = {}
        for ntype, feat in h.items():
            proj_out[ntype] = self.proj[ntype](feat.float())

        # Step 2: pass through node attention + semantic attention
        for gnn in self.layers:
            proj_out = gnn(g, proj_out)

        # Step 3: pass through MLP
        return self.predict(h2)
