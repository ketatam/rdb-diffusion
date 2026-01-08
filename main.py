"""
This script is the main entry point for this repository. It allows you to:
1. load and preprocess a relational database into a graph format,
2. train a Graph-Conditional Relational Diffusion Model (GRDM) to learn the attribute distribution conditioned on the graph structure,
3. sample a synthetic graph structure using a node degree-preserving random graph generation algorithm,
4. sample synthetic node attributes conditioned on the sampled graph structure using the trained GRDM,
5. evaluate the quality of the generated synthetic relational database using multiple metrics.

For details about the method, please refer to the paper: https://arxiv.org/abs/2505.16527
"""

import argparse
import os
from copy import deepcopy

import torch

import wandb
from rdbdiff.dataloader import get_neighbor_dataloader
from rdbdiff.dataset import RDBDiffDataset
from rdbdiff.model import GRDM, GNNDenoiser
from rdbdiff.random_graph_sampler import RandomGraphSampler
from rdbdiff.sampler import GRDMSampler
from rdbdiff.third_party.clava_ddpm import gen_multi_report
from rdbdiff.trainer import GRDMTrainer
from rdbdiff.utils import (
    count_params,
    denormalize_data,
    extract_attributes,
    get_col_to_stype_dict,
    make_pkey_fkey_graph,
    manual_seed,
    preprocess_database,
    read_json,
    read_pickle,
    round_synthetic_data,
    write_database,
    write_pickle,
)

#########################
### 0. Config Loading ###
#########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "config_path", type=str, help="Path to the dataset-specific json config file."
)
# general args
parser.add_argument(
    "--exp_name",
    type=str,
    default="default_run",
    help="Descriptive experiment name used to separate logging and model checkpoints of different experiments on the same dataset",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--fixed_structure",
    action="store_true",
    help="Whether to use the training graph structure instead of sampling a new one (for ablation)",
)
parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Whether to disable wandb logging, useful when debugging.",
)
# dataloader args
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=4096,
    help="Training batch size. Can be reduced to reduce memory requirements.",
)
parser.add_argument(
    "--n_hops",
    type=int,
    default=1,
    help="Number of hops to sample in each subgraph. Will also be used to set GNN layers.",
)
parser.add_argument(
    "--num_neighbors",
    nargs="+",
    type=int,
    default=[-1],
    help="Number of neighbors to sample at each hop. Set -1 to sample all neighbors.",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Whether to use CPU instead of GPU (default). Useful for debugging.",
)
args = parser.parse_args()

# set random seed for reproducibility
seed = args.seed
manual_seed(seed)

# extract args
# general args
config_path = args.config_path
exp_name = args.exp_name
generate_structure = not args.fixed_structure
# dataloader args
train_batch_size = args.train_batch_size
n_hops = args.n_hops
num_neighbors = args.num_neighbors
assert len(num_neighbors) == n_hops, "num_neighbors length must match n_hops"
use_cpu = args.cpu

# load config
config = read_json(config_path)
dataset_name = config["data"]["dataset_name"]
data_path = config["data"]["data_path"]
dimension_tables = config["data"]["dimension_tables"]
if len(dimension_tables) == 0:
    dimension_tables = None
# model args
gnn_aggr = config["model"]["gnn_aggr"]
d_layers = config["model"]["d_layers"]
dropout = config["model"]["dropout"]
dim_t = config["model"]["dim_t"]
# diffusion args
num_timesteps = config["diffusion"]["num_timesteps"]
gaussian_loss_type = config["diffusion"]["gaussian_loss_type"]
scheduler = config["diffusion"]["scheduler"]
# trainer args
lr = config["trainer"]["lr"]
weight_decay = config["trainer"]["weight_decay"]
steps = config["trainer"]["steps"]
# sampler args
sample_batch_size = config["sampler"]["sample_batch_size"]
size_multiplier = config["sampler"]["size_multiplier"]

# prepare workspace directory and device
workspace_dir = f"rdbdiff_workspace/{dataset_name}/{exp_name}/seed_{seed}"
os.makedirs(workspace_dir, exist_ok=True)
device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not use_cpu
    else torch.device("cpu")
)

# initialize wandb
print("Initializing W&B...")
wandb.init(
    project="RDBDiff",
    group=dataset_name,
    name=f"{dataset_name}_{exp_name}_seed_{seed}",
    mode="disabled" if args.no_wandb else None,
)

# print all args
print("General args:")
print(f"  config_path: {config_path}")
print(f"  exp_name: {exp_name}")
print(f"  seed: {seed}")
print(f"  generate_structure: {generate_structure}")
print("Dataloader args:")
print(f"  train_batch_size: {train_batch_size}")
print(f"  n_hops: {n_hops}")
print(f"  num_neighbors: {num_neighbors}")
print("Data args:")
print(f"  dataset_name: {dataset_name}")
print(f"  data_path: {data_path}")
print(f"  dimension_tables: {dimension_tables}")
print("Model args:")
print(f"  gnn_aggr: {gnn_aggr}")
print(f"  d_layers: {d_layers}")
print(f"  dropout: {dropout}")
print(f"  dim_t: {dim_t}")
print("Diffusion args:")
print(f"  num_timesteps: {num_timesteps}")
print(f"  gaussian_loss_type: {gaussian_loss_type}")
print(f"  scheduler: {scheduler}")
print("Trainer args:")
print(f"  lr: {lr}")
print(f"  weight_decay: {weight_decay}")
print(f"  steps: {steps}")
print("Sampler args:")
print(f"  sample_batch_size: {sample_batch_size}")
print(f"  size_multiplier: {size_multiplier}")
print("Other args:")
print(f"  workspace_dir: {workspace_dir}")
print(f"  device: {device}")

###########################
### 1. Data Preparation ###
###########################

# load data from disk
print("Loading data...")
dataset_real = RDBDiffDataset(
    data_path, cache_dir=None
)  # use cache_dir if needed for large data
database_real = dataset_real.get_db()
dataset_meta = dataset_real.dataset_meta

# preprocess data: normalize values using quantile transform
print("Preprocessing data...")
# TODO(AK): assert we can reconstruct all/most data
normalizers_dir = os.path.join(workspace_dir, "normalizers")
preprocess_database(database_real, normalizers_dir, seed)

# construct graph
print("Constructing graph...")
col_to_stype_dict = get_col_to_stype_dict(database_real, data_path)
graph_real = make_pkey_fkey_graph(
    database_real,
    col_to_stype_dict=col_to_stype_dict,  # speficied column types
)
# NOTE 1: make_pkey_fkey_graph removes primary and foreign key columns from col_to_stype_dict
# NOTE 2: from this point onwards, we view the data as a graph so we use graph terminology
# e.g. node_type refers to table_name etc.

# get dataloader
node_type_to_d_in = {
    node_type: graph_real[node_type].x.shape[1] for node_type in graph_real.node_types
}
train_dataloader, n_total_input_nodes = get_neighbor_dataloader(
    data=graph_real,
    batch_size=train_batch_size,
    shuffle=True,
    disjoint=True,
    num_neighbors=num_neighbors,
    dimension_tables=dimension_tables,
    node_type_to_d_in=node_type_to_d_in,
)

########################
### 2. GRDM Training ###
########################

# prepare model
gnn_params = {
    "node_types": graph_real.node_types,
    "edge_types": graph_real.edge_types,
    "aggr": gnn_aggr,
    "num_layers": n_hops,  # consistent with the number of hops in the dataloaders
}
rtdl_params = {
    "dropout": dropout,
    "d_layers": d_layers,
}
# set the number of layers in the RTDL MLPs based on the size of each table
node_type_to_rtdl_d_layers = {}
for node_type in graph_real.node_types:
    if dimension_tables is not None and node_type in dimension_tables:
        continue
    if graph_real[node_type].num_nodes >= 10000:
        node_type_to_rtdl_d_layers[node_type] = d_layers
    else:
        node_type_to_rtdl_d_layers[node_type] = [512, 1024, 512]
print(f"node_type_to_rtdl_d_layers: {node_type_to_rtdl_d_layers}")

print(f"Instantiating diffusion model...")
# denoising model
denoising_model = GNNDenoiser(
    node_type_to_d_in,
    gnn_params,
    rtdl_params,
    dim_t,
    dimension_tables=dimension_tables,
    node_type_to_rtdl_d_layers=node_type_to_rtdl_d_layers,
)
denoising_model.to(device)

# diffusion model
grdm = GRDM(
    denoise_fn=denoising_model,
    num_timesteps=num_timesteps,
    gaussian_loss_type=gaussian_loss_type,
    scheduler=scheduler,
    device=device,
    dimension_tables=dimension_tables,
)
grdm.to(device)
grdm.train()

print(f"TOTAL Number of parameters in GRDM: {count_params(grdm)}")
print(f"Number of parameters in denoising model: {count_params(denoising_model)}")
print(f"Number of parameters of GNN: {count_params(denoising_model.gnn)}")
print(f"Number of parameters of MLPs: {count_params(denoising_model.mlps)}")
print(
    f"Number of parameters of time embedding: {count_params(denoising_model.time_embed)}"
)
print(
    f"Number of parameters of input projections: {count_params(denoising_model.projs)}"
)
wandb.log({"model/num_params": count_params(grdm)["params-total"]})

# training loop
model_ckpt_dir = os.path.join(workspace_dir, "model_checkpoint")
trainer = GRDMTrainer(
    diffusion_model=grdm,
    train_dataloader=train_dataloader,
    lr=lr,
    weight_decay=weight_decay,
    steps=steps,
    save_dir=model_ckpt_dir,
    device=device,
    train_batch_size=train_batch_size,
    n_total_target_nodes=n_total_input_nodes,
    max_epochs=None,
    target_node_types=[
        nt
        for nt in graph_real.node_types
        if dimension_tables is None or nt not in dimension_tables
    ],
)

if not trainer.is_model_already_trained():
    print(f"Training model...")
    trainer.run_training_loop()
    trainer.save_model_and_loss_history()
else:
    print(f"Loading trained model...")
    trainer.load_model()

# use the EMA model for sampling
grdm._denoise_fn = trainer.ema_model
grdm.eval()

# once training is done, we can sample from the diffusion models
sampler = GRDMSampler(
    diffusion_model=grdm,
    save_dir=model_ckpt_dir,
    num_timesteps=num_timesteps,
    num_neighbors=num_neighbors,
    sample_batch_size=sample_batch_size,
    device=device,
    dimension_tables=dimension_tables,
    node_type_to_d_in=node_type_to_d_in,
)

#####################################
### 3. Graph Structure Generation ###
#####################################

# sampling
if generate_structure:
    syn_data_dir = os.path.join(
        workspace_dir,
        f"synthetic_data_sampled_structure_size_multiplier_{size_multiplier}",
    )
    os.makedirs(syn_data_dir, exist_ok=True)

    sampled_structure_dir = os.path.join(syn_data_dir, "sampled_structure")
    if not os.path.exists(sampled_structure_dir):
        print("Sampling structure...")
        structure_sampler = RandomGraphSampler(
            real_data_path=data_path,
            dimension_tables=dimension_tables,
            keep_pkey_fkeys=None,
        )
        structure_sampler.sample_and_save_structure(
            seed=seed,
            size_multiplier=size_multiplier,
            save_dir=sampled_structure_dir,
        )
    else:
        print("Sampled structure is already generated. Loading it...")

    # load in both cases
    database_syn = RDBDiffDataset(sampled_structure_dir, cache_dir=None).get_db()
    preprocess_database(
        database_syn, normalizers_dir, seed
    )  # VERY IMPORTANT when using dimension tables
    graph_empty_attributes = make_pkey_fkey_graph(
        database_syn, col_to_stype_dict=col_to_stype_dict
    )
else:  # fixed structure
    syn_data_dir = os.path.join(workspace_dir, "synthetic_data_real_structure")
    os.makedirs(syn_data_dir, exist_ok=True)

    graph_empty_attributes = deepcopy(graph_real)
    for node_type in graph_empty_attributes.node_types:
        if dimension_tables is not None and node_type in dimension_tables:
            continue
        graph_empty_attributes[node_type].x = torch.zeros_like(
            graph_empty_attributes[node_type].x
        )

################################################
### 4. Attribute Sampling and Postprocessing ###
################################################

X_syn_path = os.path.join(syn_data_dir, "node_type_to_X_syn.pkl")
if not os.path.exists(X_syn_path):
    print("Sampling...")
    graph_syn = sampler.sample_node_attributes(graph_empty_attributes)

    node_type_to_X_syn = {
        node_type: graph_syn[node_type].x.detach().cpu().numpy()
        for node_type in graph_syn.node_types
    }
    write_pickle(node_type_to_X_syn, X_syn_path)
else:
    print("Synthetic data is already sampled. Loading...")
    node_type_to_X_syn = read_pickle(X_syn_path)

# denormalize synthetic data
node_type_to_X_syn = denormalize_data(node_type_to_X_syn, normalizers_dir)

# re-load raw data (without preprocessing)
database_raw = RDBDiffDataset(data_path, cache_dir=None).get_db()

# round synthetic data
node_type_to_num_numerical_features = {
    node_type: graph_real[node_type].num_numerical_features[0].item()
    for node_type in graph_real.node_types
}
node_type_to_X_syn = round_synthetic_data(
    node_type_to_X_syn=node_type_to_X_syn,
    node_type_to_X_real=extract_attributes(database_raw),
    node_type_to_num_numerical_features=node_type_to_num_numerical_features,
)

if not generate_structure:
    database_syn = deepcopy(database_raw)  # only the structure will be kept

for table_name, table in database_syn.table_dict.items():
    df = table.df
    cols_no_id = [col for col in df.columns if not col.endswith("_id")]
    df[cols_no_id] = node_type_to_X_syn[table_name]

# for evaluation, save synthetic data to disk
database_syn_path = os.path.join(syn_data_dir, "synthetic_database")
write_database(database_syn, database_syn_path, suffix="_synthetic")

# also save real data because it needs to be sorted in the same was as synthetic data
database_real_path = os.path.join(workspace_dir, "real_database")
write_database(database_raw, database_real_path)

##################################
### 5. Data Quality Evaluation ###
##################################

# eval
print("Evaluating...")
eval_result = gen_multi_report(
    real_data_path=database_real_path,
    syn_data_path=database_syn_path,
    metadata_dir=data_path,
    eval_save_dir=syn_data_dir,
)

print(f"Final results on dataset {dataset_name}:")
print("Cardinality:", f'{eval_result["report"].get_properties()["Score"][2]*100:.04}')
print("Column Shapes:", f'{eval_result["report"].get_properties()["Score"][0]*100:.04}')
for i in range(len(eval_result["avg_scores"])):
    if i == 0:
        print("Intra-Table Trends:", f'{eval_result["avg_scores"][i]*100:.04}')
    else:
        print(f"{i}-HOP:", f'{eval_result["avg_scores"][i]*100:.04}')
print("AVG 2-WAY:", f'{eval_result["all_avg_score"]*100:.04}')

# wandb
wandb.log(
    {
        "eval/Cardinality": round(
            eval_result["report"].get_properties()["Score"][2] * 100, 2
        ),
        "eval/Column Shapes": round(
            eval_result["report"].get_properties()["Score"][0] * 100, 2
        ),
        "eval/AVG 2-WAY": round(eval_result["all_avg_score"] * 100, 2),
    }
)
for i in range(len(eval_result["avg_scores"])):
    if i == 0:
        wandb.log(
            {"eval/Intra-Table Trends": round(eval_result["avg_scores"][i] * 100, 2)}
        )
    else:
        wandb.log({f"eval/{i}-HOP": round(eval_result["avg_scores"][i] * 100, 2)})

print("Done!")
