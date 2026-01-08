# Joint Relational Database Generation via Graph-Conditional Diffusion Models
This repository provides the official implementation of the paper [Joint Relational Database Generation via Graph-Conditional Diffusion Models](https://arxiv.org/abs/2505.16527), which was published at [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/117382).

If you encounter any problem with the code or if you have any questions or suggestions, feel free to open an issue on GitHub or contact me at a.ketata@tum.de.

**Abstract**

>*Building generative models for relational databases (RDBs) is important for many applications, such as privacy-preserving data release and augmenting real datasets. However, most prior works either focus on single-table generation or adapt single-table models to the multi-table setting by relying on autoregressive factorizations and sequential generation. These approaches limit parallelism, restrict flexibility in downstream applications, and compound errors due to commonly made conditional independence assumptions. In this paper, we propose a fundamentally different approach: jointly modeling all tables in an RDB without imposing any table order. By using a natural graph representation of RDBs, we propose the Graph-Conditional Relational Diffusion Model (GRDM), which leverages a graph neural network to jointly denoise row attributes and capture complex inter-table dependencies. Extensive experiments on six real-world RDBs demonstrate that our approach substantially outperforms autoregressive baselines in modeling multi-hop inter-table correlations and achieves state-of-the-art performance on single-table fidelity metrics.*

## Setup
### Installation

1. Create a conda virtual environemnt and activate it
```sh
conda create -n rdbdiff python=3.11
conda activate rdbdiff
```

2. Clone and install the repository
```sh
# clone the repository
git clone https://github.com/ketatam/rdb-diffusion.git
cd rdb-diffusion
# Install package editably with dependencies
pip install -e .
# install additional requirements
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
# install pre-commit hooks
pre-commit install
```

### Datasets
In the paper, we use six datasets. Three of them (California, Movie Lens, and RelBench-F1) are included with this repository. The other three (Berka, Instacart 05, and CCS) can be downloaded from [this link](https://drive.google.com/file/d/1H_IGsL7JuCJYlq_6TOmdIZ4-q-wFafCL/view?usp=sharing), which comes from [ClavaDDPM's repository](https://github.com/weipang142857/ClavaDDPM). Each dataset's folder should be placed into the `data/` folder, similarly to the provided examples.

## Experiments
The `main.py` file is the main entry point to the repository. It allows you to train a model on a given database, sample a synthetic version, and evaluate the quality of the sampled database compared to the real one.

The only required argument is the path to the config file, which contains details about the database and some other model parameters. However, we highly recommend setting the `--exp_name` argument to a meaningful value that describes the main characterstics of the run. This will allow to organize different model checkpoints and caching artifacts into different folders.

Note that this repository uses [Weights & Biases](https://wandb.ai/site/) for logging, so you need to set it up first.

To launch an experiment on the California database using the default parameters, run:

```sh
python main.py configs/california.json --exp_name default_run
```

To use a different dataset, just replace the config file path with the desired one, e.g. `configs/movie_lens.json` for the Movie Lens database.

### Increasing the number of hops
The major hyperparameter of our model is the number of hops used to sample the training subgraphs and also to set the number of layers in the GNN. The default value is `1`, which works well for all used datasets with a relatively low runtime and memory consumption.

On some datasets (Berka, Instacart 05, and RelBench-F1), we observed some improvement when increasing the number of hops to `2`, at the expense of a higher runtime and memory consumption. Note that, depending on your hardware setup, you might need to decrease the training batch size to avoid out-of-memory issues when using `n_hops=2`.

To launch an experiment on the Instacart 05 database using `n_hops=2`, run:

```sh
python main.py configs/instacart_05.json --n_hops 2 --num_neighbors -1 -1 --train_batch_size 1024 --exp_name two_hops
```

### Sampling from an already-trained model
When running an experiment for the first time, the following directory is created: `f"rdbdiff_workspace/{dataset_name}/{exp_name}/seed_{seed}"`, which will contain the model's checkpoint, the generated synthetic database, and other artifacts such as data normalizers, training and evaluation results.

If you run the same experiment again (same dataset, same exp_name and same seed), the training loop is skipped and the already-trained model is loaded instead. This is helpful if the previous run crashed during sampling or if you want to use the same model for sampling a new synthetic database with different sampling configurations.

For instance, while the default configuration samples a new graph structure for the synthetic database, it might be interesting to use the real structure and only sample synthetic attributes. To do this, run:
```sh
python main.py configs/california.json --exp_name default_run --fixed_structure
```
Note that this will only skip the training part if the model has already been trained, otherwise it will train it from scratch.

Another sampling hyperparameter which can be changed to sample from the same trained model is the `size_multiplier`, which controls the size of the synthetic database compared to the real database size. It can be changed directly from the config file.

Note that if you run the same command with the same sampling hyperparameters twice, the sampling will also be skipped and only the evaluation is performed again. If you want to sample again, you need to manually delete the folder containing the sampled database (e.g. `synthetic_data_sampled_structure_size_multiplier_1.0` or `synthetic_data_real_structure`).

### Using you own database
We are currently working on cleaning up the code for data preprocessing and will release it as soon as possible into this repository.

In the meantime, if you want to run our code on new datasets, you need to process them into the same format as the provided examples. The rough steps for this should be:
1. All tables of the RDB should be saved as .csv files with the table name serving as the file name. The primary key of all tables should be of the form `<table_name>_id`. Similarly, the foreign keys should be of the form `<other_table_name>_id`, where `<other_table_name>` is a valid name of another table of the RDB. The primary keys of a table should be unique. All other columns should be converted into numerical format (or discarded if this is not possible, e.g. for free-form text as this is not yet supported). Concretely, categorical columns should be mapped to integer values {0,1,2,...}, date/time columns should be converted into numerical values, e.g., by computing the relative time (in seconds or days, etc.) to the earliest timestamp in the table. Numerical columns should be kept as numerical columns.
2. Create .json files of the form `<table_name>_domain.json` that specify the type (discrete or continuous) and cardinality of every columns in the table.
3. Finally, create a `dataset_meta.json` file that specifies the relationship between the tables. Specficially, `relation_order` is topological order traveral of the RDB schema graph and `tables` gives the parents and children of each table.

[This file](https://github.com/weipang142857/ClavaDDPM/blob/main/preprocess_utils.py) might be helpful to implement these steps.

## Citation
If you build upon this work, please consider citing our paper:
```
@inproceedings{
ketata2025joint,
title={Joint Relational Database Generation via Graph-Conditional Diffusion Models},
author={Mohamed Amine Ketata and David L{\"u}dke and Leo Schwinn and Stephan G{\"u}nnemann},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Z3OtNSwuXX}
}
```
