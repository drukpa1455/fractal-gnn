# FGNNs
Are you familiar with fractal graph neural networks (FGNNs)? FGNNs are a type of graph neural network (GNN) architecture that incorporates concepts from fractal theory to capture multi-scale hierarchical structures in graph-structured data.

## Key characteristics of FGNNs:
- Fractal structure: FGNNs model graphs as self-similar structures, where subgraphs at different scales exhibit similar patterns and properties.
- Hierarchical representation: FGNNs learn hierarchical representations of graphs by applying graph convolution operations at multiple scales or resolutions.
- Recursive aggregation: The network aggregates node features recursively across different scales, allowing information to flow from local to global contexts.
- Efficient computation: By exploiting the self-similarity and sparsity of fractal graphs, FGNNs can perform efficient computations and reduce the computational complexity compared to traditional GNNs.

FGNNs have been applied to various tasks, such as graph classification, node classification, and link prediction. They have shown promising results in capturing multi-scale patterns and long-range dependencies in graph-structured data.

## Some notable works on FGNNs include:
1. "Fractal Graph Neural Networks" by Zhiqian Chen et al. (2019)
2. "Fractional Graph Convolutional Networks" by Zhiqian Chen et al. (2020)
3. "Fractal Graph Attention Networks" by Zhiqian Chen et al. (2020)

FGNNs are an active area of research, and there have been further developments and variations of the original architecture to address specific challenges and improve performance on various graph-related tasks.

# FGNN
The mathematical formulation of FGNNs can be summarized as follows:

1. Fractal graph generation: Given an input graph G, the authors generate a fractal graph G_f by recursively applying a graph expansion operation. This operation replaces each node in the original graph with a subgraph, creating a hierarchical structure.
2. Fractal graph convolution: The fractal graph convolution operation is defined as:
```h_i^(l+1) = σ(W_l ∑_j∈N_i h_j^l + b_l)```
where h_i^(l+1) is the feature vector of node i at layer l+1, W_l and b_l are learnable parameters, N_i is the neighborhood of node i, and σ is a non-linear activation function.
3. Hierarchical aggregation: The node features are aggregated hierarchically across different scales of the fractal graph. The aggregated features at each scale are combined using a weighted sum:
```h_i = ∑_{l=1}^L α_l h_i^l```
where h_i is the final feature representation of node i, L is the number of scales, and α_l are learnable scale-specific weights.