from typing import Dict

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import NodeType


# NOTE: it might happen that input_id gets detected as an edge feature when running batch.to_heterogeneous()
# keep this in mind in case of a future error as this might be the reason.
def get_neighbor_dataloader(
    data: HeteroData,
    batch_size: int = 1,
    shuffle: bool = False,
    disjoint: bool = True,
    num_neighbors: list = None,
    dimension_tables: list = None,
    node_type_to_d_in: Dict[NodeType, int] = None,
) -> NeighborLoader:
    r"""Return a neighbor dataloader for a given database.

    When using PyG's NeighborLoader on heterogeneous graphs, only one node type can be used as
    seed nodes for sampling. To overcome this limitation, we first convert the heterogeneous
    graph to a homogeneous graph, and then use NeighborLoader to sample subgraphs which then contain
    seed nodes of all types.
    The sampled subgraphs are then converted back to heterogeneous graphs using the make_heterogeneous
    function passed as a transform to the NeighborLoader.
    """
    data_homogeneous = data.to_homogeneous()
    # nodes from dimension tables should not be used as seed nodes
    if dimension_tables is not None:
        dimension_tables_indices = [
            data_homogeneous._node_type_names.index(node_type)
            for node_type in dimension_tables
        ]
        # input_nodes is a boolean mask indicating which nodes can be used as seed nodes (True if not in dimension tables, False otherwise)
        input_nodes = torch.isin(
            data_homogeneous.node_type,
            torch.Tensor(dimension_tables_indices),
            invert=True,
        )
        n_total_input_nodes = input_nodes.sum().item()
    else:
        input_nodes = None
        n_total_input_nodes = data_homogeneous.num_nodes
    # requires pyg-lib. can be installed with e.g. `pip install pyg-lib -f https://data.pyg.org/whl/torch-2.5.1+cu124.html`
    # nice Doc: https://pytorch-geometric.readthedocs.io/en/2.6.1/modules/loader.html#torch_geometric.loader.NeighborLoader
    data_loader = NeighborLoader(
        data_homogeneous,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        persistent_workers=False,
        drop_last=False,
        disjoint=disjoint,  # VERY important in order to add the same noise level for each seed node's neighbors during the forward diffusion process
        transform=lambda x: make_heterogeneous(x, node_type_to_d_in),
    )
    return data_loader, n_total_input_nodes


# NOTE: num_sampled_nodes, num_sampled_edges, input_id, and batch_size need to be mapped to the different node types, if needed
def make_heterogeneous(batch, node_type_to_d_in):
    # make the graph heterogeneous
    batch_hetero = batch.to_heterogeneous()
    # sometimes, the input_id field gets automatically detected as an edge feature when running batch.to_heterogeneous()
    # this happens if the number of sampled edges matches the number of nodes somehow
    # for now a hacky solution is to simply create the input_id field again if it does not exist
    # TODO: find a more elegant solution
    try:
        _ = batch_hetero.input_id
    except AttributeError:
        batch_hetero.input_id = batch.input_id
    # remove the extra dimensions that were padded when the graph was made homogeneous
    for node_type in batch_hetero.node_types:
        num_features = node_type_to_d_in[node_type]
        batch_hetero[node_type].x = batch_hetero[node_type].x[:, :num_features]
    return batch_hetero
