import json
import os
import pickle
from typing import Dict, Union

import numpy as np
import sklearn.preprocessing
from pandas import DataFrame
from relbench.base import Database
from scipy.spatial.distance import cdist


def read_json(path: str):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data


def read_pickle(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def sort_columns(
    df: DataFrame, domain: Dict[str, Dict[str, Union[str, int]]]
) -> DataFrame:
    """Sort columns of a DataFrame such that ID columns are first, then numerical features, then categorical features."""
    numerical_columns, categorical_columns = [], []
    for col, info in domain.items():
        if info["type"] == "continuous":
            numerical_columns.append(col)
        elif info["type"] == "discrete":
            categorical_columns.append(col)
        else:
            raise ValueError(f"Unknown feature type {info['type']}.")

    sorted_columns = (
        list(set(df) - set(numerical_columns) - set(categorical_columns))  # id columns
        + numerical_columns  # then numerical columns
        + categorical_columns  # then categorical columns
    )
    return df[sorted_columns]


def get_col_to_stype_dict(
    database: Database,
    data_path: str,
) -> Dict[str, Dict[str, str]]:
    type_to_stype = {
        "discrete": "categorical",
        "continuous": "numerical",
    }
    col_to_stype_dict = {}
    for table_name, table in database.table_dict.items():
        domain_path = os.path.join(data_path, f"{table_name}_domain.json")
        domain = read_json(domain_path)
        col_to_stype_dict[table_name] = {}
        for col in table.df.columns:
            if col.endswith("_id"):
                # TODO: should it be categorical or numerical?
                col_to_stype_dict[table_name][col] = "categorical"
            else:
                col_to_stype_dict[table_name][col] = type_to_stype[domain[col]["type"]]

    return col_to_stype_dict


# TODO: generalize to more normalization functions
# for now, we only need QuantileTransformer, so it's fine
def preprocess_database(
    database: Database,
    normalizers_cache_dir: str,
    seed: int = 0,
):
    """Preprocess a database by normalizing its data in-place."""
    for table_name, table in database.table_dict.items():
        df = table.df
        cols_no_id = [col for col in df.columns if not col.endswith("_id")]
        data_raw = df[cols_no_id].to_numpy()
        data_processed = normalize(
            X=data_raw,
            cache_dir=normalizers_cache_dir,
            table_name=table_name,
            seed=seed,
        )
        df[cols_no_id] = data_processed


def normalize(
    X: np.ndarray,
    cache_dir: str,
    table_name: str,
    seed: int = 0,
):
    os.makedirs(cache_dir, exist_ok=True)
    normalizer_path = os.path.join(cache_dir, f"{table_name}_normalizer.pkl")

    if os.path.exists(normalizer_path):
        # normalizer has already been fitted
        print(f"Using cached normalizer for table {table_name}.")
        normalizer = read_pickle(normalizer_path)
    else:
        print(f"Fitting normalizer from scratch for table {table_name}.")
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        normalizer.fit(X)
        write_pickle(normalizer, normalizer_path)

    X_transformed = normalizer.transform(X)
    return X_transformed


def denormalize_data(
    node_type_to_X_syn: Dict[str, np.ndarray],
    normalizers_dir: str,
):
    node_type_to_X_denorm = {}
    for node_type, X_syn in node_type_to_X_syn.items():
        X_denorm = denormalize(
            X=X_syn,
            cache_dir=normalizers_dir,
            table_name=node_type,
        )
        node_type_to_X_denorm[node_type] = X_denorm
    return node_type_to_X_denorm


def denormalize(
    X: np.ndarray,
    cache_dir: str,
    table_name: str,
):
    normalizer_path = os.path.join(cache_dir, f"{table_name}_normalizer.pkl")
    if not os.path.exists(normalizer_path):
        raise ValueError(f"Normalizer for table {table_name} not found.")
    print(f"Using cached normalizer for table {table_name}.")
    normalizer = read_pickle(normalizer_path)
    X_denormalized = normalizer.inverse_transform(X)
    return X_denormalized


def extract_attributes(database: Database) -> Dict[str, np.ndarray]:
    table_name_to_X = {}
    for table_name, table in database.table_dict.items():
        df = table.df
        cols_no_id = [col for col in df.columns if not col.endswith("_id")]
        table_name_to_X[table_name] = df[cols_no_id].to_numpy()
    return table_name_to_X


def round_synthetic_data(
    node_type_to_X_syn: Dict[str, np.ndarray],
    node_type_to_X_real: Dict[str, np.ndarray],
    node_type_to_num_numerical_features: Dict[str, int],
) -> Dict[str, np.ndarray]:
    for table_name in node_type_to_X_syn:
        X_syn = node_type_to_X_syn[table_name]
        X_real = node_type_to_X_real[table_name]
        num_numerical_features = node_type_to_num_numerical_features[table_name]

        X_syn_num = X_syn[:, :num_numerical_features]
        X_syn_cat = X_syn[:, num_numerical_features:]

        X_real_num = X_real[:, :num_numerical_features]
        X_real_cat = X_real[:, num_numerical_features:]

        # round categorical features
        X_syn_cat = np.round(X_syn_cat).astype(int)
        # TODO: clip between smallest and largest class label
        # TODO: using sklear.LabelEncoder might be helpful for this purpose as it stores the number of classes
        # for now just check and manually correct
        for col in range(X_syn_cat.shape[1]):
            real_unique = np.unique(X_real_cat[:, col])
            assert (real_unique == np.arange(len(real_unique))).all()
            X_syn_cat[:, col] = np.clip(X_syn_cat[:, col], 0, len(real_unique) - 1)

            syn_unique = np.unique(X_syn_cat[:, col])
            if (
                len(syn_unique) != len(real_unique)
                or not (syn_unique == real_unique).all()
            ):
                print(
                    f"A categorical column's unique values do not match between synthetic and real data for table {table_name}, column {col}"
                )

        # round numerical features
        # X_syn_num = X_syn_num.astype(int)
        X_syn_num = np.round(X_syn_num, 6)

        disc_cols = []
        for col in range(X_real_num.shape[1]):
            uniq_vals = np.unique(X_real_num[:, col])
            # in clava it is: if len(uniq_vals) <= 32
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print(
            "Discrete cols (<= 32 integer categories) modeled as numerical cols:",
            disc_cols,
        )
        if len(disc_cols):
            X_syn_num = round_columns(X_real_num, X_syn_num, disc_cols)

        for col in range(X_syn_num.shape[1]):
            syn_unique = np.unique(X_syn_num[:, col])
            real_unique = np.unique(X_real_num[:, col])
            if (
                len(syn_unique) != len(real_unique)
                or not (syn_unique == real_unique).all()
            ):
                print(
                    f"A numerical column's unique values do not match between synthetic and real data for table {table_name}, column {col}"
                )

        node_type_to_X_syn[table_name] = np.concatenate((X_syn_num, X_syn_cat), axis=1)

    return node_type_to_X_syn


def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:, col])
        dist = cdist(
            X_synth[:, col][:, np.newaxis].astype(float),
            uniq[:, np.newaxis].astype(float),
        )
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth


def index_columns(df: DataFrame) -> DataFrame:
    """Index columns of a DataFrame such that each column is prefixed with its index."""
    cols_no_id = [col for col in df.columns if not col.endswith("_id")]
    num_digits = len(str(len(cols_no_id) - 1))
    col_name_mapper = {
        col: f"{i:0{num_digits}d}_{col}" for i, col in enumerate(cols_no_id)
    }
    return df.rename(columns=col_name_mapper)


def get_feature_types_count(db: Database, data_path: str):
    feature_type_to_count_dict = {}
    for table_name in db.table_dict:
        feature_type_to_count_dict[table_name] = {
            "numerical": 0,
            "categorical": 0,
        }
        domain_path = os.path.join(data_path, f"{table_name}_domain.json")
        domain = read_json(domain_path)
        for col in domain:
            if domain[col]["type"] == "continuous":
                feature_type_to_count_dict[table_name]["numerical"] += 1
            elif domain[col]["type"] == "discrete":
                feature_type_to_count_dict[table_name]["categorical"] += 1
            else:
                raise ValueError(f"Unknown feature type {domain[col]['type']}.")
    return feature_type_to_count_dict


def write_database(
    database: Database,
    save_dir: str,
    suffix: str = "",
):
    os.makedirs(save_dir, exist_ok=True)
    for table_name, table in database.table_dict.items():
        table.df.to_csv(
            os.path.join(save_dir, f"{table_name}{suffix}.csv"), index=False
        )
