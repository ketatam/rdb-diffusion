from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from relbench.modeling.nn import HeteroGraphSAGE
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from rdbdiff.third_party.tab_ddpm import (
    MLP,
    GaussianMultinomialDiffusion,
    timestep_embedding,
)


# The Graph-Conditional Relational Diffusion Model (GRDM)
class GRDM(GaussianMultinomialDiffusion):
    def __init__(
        self,
        denoise_fn,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        gaussian_parametrization="eps",
        multinomial_loss_type="vb_stochastic",
        parametrization="x0",
        scheduler="cosine",
        device=torch.device("cpu"),
        dimension_tables: list = None,
    ):
        super(GRDM, self).__init__(
            num_classes=np.array([0]),
            num_numerical_features=0,
            denoise_fn=denoise_fn,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            gaussian_parametrization=gaussian_parametrization,
            multinomial_loss_type=multinomial_loss_type,
            parametrization=parametrization,
            scheduler=scheduler,
            device=device,
        )
        self.device = device
        self.dimension_tables = dimension_tables

    def relational_gaussian_loss(
        self,
        batch: HeteroData,
    ):
        batch_size = (
            batch.batch_size
        )  # number of sampled seed nodes, forming disconnected subgraphs (in case of disjoint=True)

        # sample time for batch_size target nodes/subgraphs
        t, _ = self.sample_time(batch_size, self.device, "uniform")

        # clean features
        x_dict = batch.x_dict  # {node_type: Tensor(num_nodes, d_in)}

        # noisy features
        x_in = {}  # {node_type: Tensor(num_nodes, d_in)}
        # noise levels
        t_in = {}  # {node_type: Tensor(num_nodes,)}
        # target features (mixture of features at noise level 0 and t-1), only needed for the batch_size target nodes
        x_target = (
            {}
        )  # {node_type: {"x_num": Tensor(num_target_nodes_node_type, d_in), "x_num_t": ..., "noise": ...}}
        for node_type in x_dict:
            # subgraph_ids[i] is the id of the target node that defines the subgraph
            # we use it to make sure that all nodes in a subgraph are noised at the same noise level
            subgraph_ids = batch[node_type].batch

            t_broadcasted = t[
                subgraph_ids
            ]  # if 2 nodes are in the same subgraph, they will have the same t

            if self.dimension_tables is not None and node_type in self.dimension_tables:
                # dimension_tables are not diffused, so we keep them as-is
                x_in[node_type] = x_dict[node_type]
                t_in[node_type] = torch.zeros_like(t_broadcasted)
                # nodes of this type are never target nodes
                continue

            # we view all features as numerical and apply Gaussian diffusion
            x_num = x_dict[node_type]

            x_num_t = x_num.clone()
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t_broadcasted, noise=noise)

            x_in[node_type] = x_num_t
            t_in[node_type] = t_broadcasted

            # collect features needed for loss computation
            # we only need the features of the input nodes, which are the first nodes of each node type
            # n_id is the global node index for every sampled node in the graph
            # input_id is the global index of the input_nodes
            # we compute the number of nodes in input_id that are also in n_id of this specific type:
            n_input_nodes = (
                torch.isin(batch.input_id, batch[node_type].n_id).sum().item()
            )

            if (
                n_input_nodes > 0
            ):  # not necessarily all node types have been sampled in this batch as input nodes
                # check that the first n_input_nodes nodes are actually input nodes:
                assert torch.isin(
                    batch[node_type].n_id[:n_input_nodes], batch.input_id
                ).all()
                x_target_node_type = {}
                x_target_node_type["t"] = t_broadcasted[:n_input_nodes]
                x_target_node_type["x_num"] = x_num[:n_input_nodes]
                x_target_node_type["x_num_t"] = x_num_t[:n_input_nodes]
                x_target_node_type["noise"] = noise[:n_input_nodes]

                x_target[node_type] = x_target_node_type

        # make sure that the total number of input nodes matches the batch size
        assert (
            np.sum([x_target[node_type]["x_num"].size(0) for node_type in x_target])
            == batch_size
        )

        # run denoising model
        model_out = self._denoise_fn(
            x_in=x_in,
            t_in=t_in,
            batch=batch,
        )  # {node_type: Tensor(num_target_nodes_node_type, d_in_node_type)}

        # compute loss
        node_type_to_loss = {}
        for node_type in model_out:
            model_out_node_type = model_out[node_type]
            x_target_node_type = x_target[node_type]

            model_out_num = model_out_node_type

            loss_gauss = self._gaussian_loss(
                model_out_num,
                x_target_node_type["x_num"],
                x_target_node_type["x_num_t"],
                x_target_node_type["t"],
                x_target_node_type["noise"],
            )  # shape: (num_target_nodes_node_type,)

            node_type_to_loss[node_type] = loss_gauss

        # NOTE: using sum for now to avoid introducing a scaling constant that would depend
        # on the number of sampled node types in this specific batch.
        loss = torch.stack(
            [loss_node_type.mean() for loss_node_type in node_type_to_loss.values()]
        ).sum()
        return loss, node_type_to_loss


class GNNDenoiser(nn.Module):
    def __init__(
        self,
        node_type_to_d_in: Dict[NodeType, int],
        gnn_params: Dict,
        rtdl_params: Dict,
        dim_t: int = 128,
        dimension_tables: list = None,
        node_type_to_rtdl_d_layers: Dict[NodeType, list] = None,
    ):
        # TODO: consider ID awareness (see paper and relbench tutorial train_model.ipynb)
        super().__init__()
        self.dim_t = dim_t

        # same embedding module is applied to the time step of every node in the graph
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

        # a different projection layer is applied to the nodes from each type
        self.projs = nn.ModuleDict()
        for node_type, d_in in node_type_to_d_in.items():
            self.projs[node_type] = nn.Linear(d_in, dim_t)

        # a single GNN is used to process the whole graph
        # specifically, during training, the GNN is used to get node representation of the input nodes.
        self.gnn = HeteroGraphSAGE(
            channels=dim_t,
            **gnn_params,
        )

        # a different MLP is applied to the representations of the input nodes obtained by the GNN
        self.mlps = nn.ModuleDict()
        for node_type, d_in in node_type_to_d_in.items():
            if dimension_tables is not None and node_type in dimension_tables:
                continue
            if node_type_to_rtdl_d_layers is not None:
                rtdl_params["d_layers"] = node_type_to_rtdl_d_layers[node_type]
            rtdl_params["d_in"] = dim_t
            rtdl_params["d_out"] = d_in
            self.mlps[node_type] = MLP.make_baseline(**rtdl_params)

        # reset parameters
        self.gnn.reset_parameters()

    def forward(
        self,
        x_in: Dict[NodeType, torch.Tensor],  # shape: (num_nodes, d_in)
        t_in: Dict[NodeType, torch.Tensor],  # shape: (num_nodes,)
        batch: HeteroData,
    ):
        # embed time
        t_emb = {
            node_type: self.time_embed(
                timestep_embedding(t, self.dim_t)
            )  # shape: (num_nodes, dim_t)
            for node_type, t in t_in.items()
        }

        # project input features and add time embedding
        x_in = {
            node_type: self.projs[node_type](x)
            + t_emb[node_type]  # shape: (num_nodes, dim_t)
            for node_type, x in x_in.items()
        }

        # run GNN
        x_out = self.gnn(
            x_in,
            batch.edge_index_dict,
        )  # {node_type: Tensor(num_nodes, dim_t)}

        # run MLPs
        x_final = {}
        for node_type in x_out:
            # extract features of input nodes
            # we only need the features of the input nodes, which are the first nodes of each node type
            # n_id is the global node index for every sampled node in the graph
            # input_id is the global index of the input_nodes
            # we compute the number of nodes in input_id that are also in n_id of this specific type:
            n_input_nodes = (
                torch.isin(batch.input_id, batch[node_type].n_id).sum().item()
            )
            if n_input_nodes > 0:
                # check that the first n_input_nodes nodes are actually input nodes:
                assert torch.isin(
                    batch[node_type].n_id[:n_input_nodes], batch.input_id
                ).all()
                input_nodes_gnn_representations = x_out[node_type][:n_input_nodes]
                x_final[node_type] = self.mlps[node_type](
                    input_nodes_gnn_representations
                )

        # make sure that the total number of input nodes matches the batch size
        assert (
            np.sum([x_final[node_type].size(0) for node_type in x_final])
            == batch.batch_size
        )
        return x_final
