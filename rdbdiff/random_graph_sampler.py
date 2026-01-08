import os
import shutil
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd

from rdbdiff.dataset import RDBDiffDataset


# Node Degree-Preserving Random Graph Generation
class RandomGraphSampler:
    def __init__(
        self,
        real_data_path: str,
        dimension_tables: list = None,
        keep_pkey_fkeys: list = None,
    ):
        self.real_data_path = real_data_path
        self.dimension_tables = dimension_tables
        self.keep_pkey_fkeys = keep_pkey_fkeys

        real_dataset = RDBDiffDataset(real_data_path, cache_dir=None)
        real_database = real_dataset.get_db()

        self.dataset_meta = real_dataset.dataset_meta
        self.real_table_to_df = {
            table_name: table.df
            for table_name, table in real_database.table_dict.items()
        }
        self.real_table_to_structure = self.get_structure_from_db(real_database)

    @staticmethod
    def get_structure_from_db(db):
        """
        Removes all feature columns from all tables in the database,
        and only keep primary and foreign key columns, which define the structure
        """
        table_to_structure = {}
        for table_name, table in db.table_dict.items():
            df = table.df
            cols_no_id = [col for col in df.columns if not col.endswith("_id")]
            table_to_structure[table_name] = df.drop(columns=cols_no_id)
        return table_to_structure

    def sample_and_save_structure(
        self,
        seed: int = None,
        size_multiplier: float = 1.0,
        root_table_to_size: dict = None,
        save_dir: str = None,
    ):
        self.sample_structure(seed, size_multiplier, root_table_to_size)
        self.initialize_features_for_sampled_structure()
        if save_dir is not None:
            self.save_generated_graph(save_dir)

    def sample_structure(
        self,
        seed: int = None,
        size_multiplier: float = 1.0,
        root_table_to_size: dict = None,
    ):
        self.add_in_degrees_to_df()
        self.compute_in_degree_distribution()
        self.check_uniqueness_fkeys()

        self.synthetic_table_to_structure = {}
        rng = np.random.default_rng(seed=seed)
        # traversal of the database graph in toplogical order
        for relation_order_parent, relation_order_child in self.dataset_meta[
            "relation_order"
        ]:
            if (
                self.dimension_tables is not None
                and relation_order_child in self.dimension_tables
            ):
                dimension_table = relation_order_child
                print(f"Using real in-degrees for dimension table {dimension_table}.")
                self.synthetic_table_to_structure[dimension_table] = deepcopy(
                    self.real_table_to_structure[dimension_table]
                )
            elif relation_order_parent is None:
                # this is a root table that is not a dimension table
                # we sample N in-degrees for each different child table
                # N is either given in root_table_to_size or by size_multiplier * real_size
                # sampling in-degrees for different child tables can be done jointly (for the same parent table, not implemented) or seperately
                # this is all we need to define the parent table but will be later used in the child table (matching in multi-parent case)
                # assumption: every root table has at least one child table, otherwise we're back to the single-table case.
                root_table = relation_order_child
                print(f"Generating root table {root_table}...")
                # the size of a root table can be directly specified or through a multiplier of the real size
                # we only need the size of root tables, because the size of all other tables will be deduced
                # from their parents and their sampled in-degrees.
                if root_table_to_size is not None:
                    # add try-except block for case when root_table is not an existing key
                    N = root_table_to_size[root_table]
                else:
                    N = round(
                        self.real_table_to_structure[root_table].shape[0]
                        * size_multiplier
                    )

                child_to_sampled_in_degrees = {}
                # NOTE: we can sample joint distributions of in-degrees.
                for child, (
                    in_degrees_unique,
                    in_degrees_frequency,
                ) in self.table_to_in_degree_distribution[root_table].items():
                    child_to_sampled_in_degrees[f"{child}_in_degree"] = rng.choice(
                        in_degrees_unique,
                        size=N,
                        replace=True,
                        p=in_degrees_frequency
                        / np.sum(in_degrees_frequency, dtype=float),
                        shuffle=False,
                    )
                df = pd.DataFrame.from_dict(
                    child_to_sampled_in_degrees, orient="columns"
                )
                df[f"{root_table}_id"] = np.arange(len(df))
                self.synthetic_table_to_structure[root_table] = df
            else:
                # we are in the case of a child table (that is not a dimension table)
                # the size of this table will be determined from the sampled in-degrees of its parent(s)
                # if it has a single parent, we directly get the fkey column from the in-degrees.
                # if it has more than one parent, we need to match these parents
                # and in case the original table contains unique fkeys, we also ensure this by re-sampling
                # more complicated matching strategies are possible, e.g. by sampling joint degrees of the parents
                # in the case of multiple parents, we will encounter the child table more than once,
                # but only process it once (the first time we see it in our traversal).
                # in both cases, we need to create a pkey column for the child table
                # finally, we sample in-degrees of potential children of this table -- exactly as done in the case of root tables

                # parent has already been processed and relation_order_child plays the role of the child table here.
                child_table = relation_order_child
                if child_table in self.synthetic_table_to_structure:
                    # multi-parent case AND already processed
                    continue
                print(f"Generating (child) table {child_table}...")
                parents = self.dataset_meta["tables"][child_table]["parents"]
                assert len(parents) > 0  # child table needs to have parents
                if (
                    self.keep_pkey_fkeys is not None
                    and child_table in self.keep_pkey_fkeys
                ):
                    print(
                        f"Using pkey and fkeys from real table for table {child_table}."
                    )
                    fkeys = {
                        f"{parent}_id": self.real_table_to_structure[child_table][
                            f"{parent}_id"
                        ].to_numpy()
                        for parent in parents
                    }
                    pkey = self.real_table_to_structure[child_table][
                        f"{child_table}_id"
                    ].to_numpy()
                    N = len(pkey)
                elif len(parents) == 1:
                    # simple single-parent case
                    # one fkey col: enumeration of parent's pkey, each in-degree times
                    # one pkey col
                    parent = parents[0]
                    # parent table is already sampled because of topological order traversal
                    parent_df = self.synthetic_table_to_structure[parent]
                    fkey_col = np.repeat(
                        parent_df[f"{parent}_id"].to_numpy(),
                        parent_df[f"{child_table}_in_degree"].to_numpy(),
                    )
                    assert fkey_col.size == parent_df[f"{child_table}_in_degree"].sum()
                    fkeys = {f"{parent}_id": fkey_col}
                    N = fkey_col.shape[0]
                    pkey = np.arange(N)
                else:  # len(parents) > 1
                    # multi-parent case
                    parents_stubs = []
                    for parent in parents:
                        # parent table is already sampled because of topological order traversal
                        parent_df = self.synthetic_table_to_structure[parent]
                        parent_stubs = np.repeat(
                            parent_df[f"{parent}_id"].to_numpy(),
                            parent_df[f"{child_table}_in_degree"].to_numpy(),
                        )
                        assert (
                            parent_stubs.size
                            == parent_df[f"{child_table}_in_degree"].sum()
                        )
                        parents_stubs.append(parent_stubs)
                    # TODO: how to deal with the different size of stubs
                    # TODO: try average size and oversample and subsample accordingly
                    n_fkeys = min(stubs.size for stubs in parents_stubs)
                    fkeys_sampled_all = None
                    n_fkeys_remaining = n_fkeys
                    parents_stubs_remaining = deepcopy(parents_stubs)
                    # TODO: only do re-sampling if original table has unique fkeys
                    # (add this check only if some tables have non-unique fkeys, which is not the case for our benchmark datasets)
                    # NOTE: we can match based on joint in-degree distributions of parents.
                    while n_fkeys_remaining > 0:
                        print(f"n_fkeys_remaining: {n_fkeys_remaining}")
                        for stubs in parents_stubs_remaining:
                            rng.shuffle(stubs)
                        # random matching:
                        fkeys_sampled = np.stack(
                            [
                                stubs[:n_fkeys_remaining]
                                for stubs in parents_stubs_remaining
                            ],
                            axis=1,
                        )
                        if fkeys_sampled_all is None:
                            fkeys_sampled_all = fkeys_sampled
                        else:
                            fkeys_sampled_all = np.concat(
                                (fkeys_sampled_all, fkeys_sampled), axis=0
                            )
                        # TODO: if not unique in the real data, skip next line
                        fkeys_sampled_all = np.unique(
                            fkeys_sampled_all, axis=0
                        )  # make a set ??
                        if n_fkeys_remaining == n_fkeys - fkeys_sampled_all.shape[0]:
                            # TODO: maybe wait a bit longer..
                            print(
                                f"Did not sample any new fkey. Stopping the generation for table {child_table}"
                            )
                            break
                        n_fkeys_remaining = n_fkeys - fkeys_sampled_all.shape[0]
                        # removing used stubs
                        for i in range(len(parents_stubs_remaining)):
                            parents_stubs_remaining[i] = self.listdiff(
                                parents_stubs[i], fkeys_sampled_all[:, i]
                            )
                    fkeys = {
                        f"{parents[i]}_id": fkeys_sampled_all[:, i]
                        for i in range(len(parents))
                    }
                    N = fkeys_sampled_all.shape[0]
                    pkey = np.arange(N)

                # create df with fkeys and pkey cols
                df = pd.DataFrame.from_dict(fkeys, orient="columns")  # fkeys
                df[f"{child_table}_id"] = pkey  # pkey
                # grandchildren in-degrees if existent:
                # NOTE: we can condition on the in-degree of the parent.
                for grandchild, (
                    in_degrees_unique,
                    in_degrees_frequency,
                ) in self.table_to_in_degree_distribution[child_table].items():
                    df[f"{grandchild}_in_degree"] = rng.choice(
                        in_degrees_unique,
                        size=N,
                        replace=True,
                        p=in_degrees_frequency
                        / np.sum(in_degrees_frequency, dtype=float),
                        shuffle=False,
                    )
                self.synthetic_table_to_structure[child_table] = df

    def add_in_degrees_to_df(self):
        """
        Computes the in-degrees for each table that has children.
        The in-degree is computed as the number of times a row in the parent table
        is referred to from the child table.
        The in-degrees are stored as an additional column `<child_name>_in_degree` inside the parent table.
        """
        for parent, metadata in self.dataset_meta["tables"].items():
            for child in metadata["children"]:
                # in-degree is the number of occurences of a row from the parent table in the child table
                parent_ids, in_degrees = np.unique(
                    self.real_table_to_structure[child][f"{parent}_id"],
                    return_counts=True,
                )
                # add nodes that have in_degreee = 0
                parent_ids_zero_in_degree = np.setdiff1d(
                    self.real_table_to_structure[parent][f"{parent}_id"], parent_ids
                )
                if len(parent_ids_zero_in_degree) > 0:
                    parent_ids = np.concat((parent_ids, parent_ids_zero_in_degree))
                    in_degrees = np.concat(
                        (in_degrees, np.zeros_like(parent_ids_zero_in_degree))
                    )

                parent_id_to_in_degree = dict(
                    zip(parent_ids.tolist(), in_degrees.tolist())
                )
                # store in-degrees as a new column in the database
                self.real_table_to_structure[parent][f"{child}_in_degree"] = (
                    self.real_table_to_structure[parent][f"{parent}_id"].map(
                        parent_id_to_in_degree
                    )
                )

    def compute_in_degree_distribution(self):
        """
        Computes the possible in-degrees from a child table to a parent table and their frequencies,
        which are then used for sampling new structures.
        Assumes `self.add_in_degrees_to_df()` has been called before.
        """
        self.table_to_in_degree_distribution = {}
        for parent, metadata in self.dataset_meta["tables"].items():
            self.table_to_in_degree_distribution[parent] = {}
            # NOTE 1: we can also compute the joint distribution of in-degrees across all children (not implemented)
            # NOTE 2: we can also condition on the degree of the parent (not implemented)
            for child in metadata["children"]:
                in_degrees_unique, in_degrees_frequency = np.unique(
                    self.real_table_to_structure[parent][f"{child}_in_degree"],
                    return_counts=True,
                )
                # TODO: interpolate missing in degree values, because intuitively they should be possible
                # simple idea: average frequency of closest existing values
                # TODO 2: potentially conditional statistics, conditioned on the degree of the parent
                self.table_to_in_degree_distribution[parent][child] = (
                    in_degrees_unique,
                    in_degrees_frequency,
                )
            # NOTE 3: we can also compute the distribution of joint parent in-degrees for a child table
            # e.g. for ratings, what is the distribution of users and movies that are connected through a rating
            # (not implemented)

    def check_uniqueness_fkeys(self):
        self.has_unique_fkeys = {}
        for table, df in self.real_table_to_structure.items():
            fkey_cols = [
                col
                for col in df.columns
                if col.endswith("_id") and col != f"{table}_id"
            ]
            if len(fkey_cols) == 0:
                self.has_unique_fkeys[table] = (0, None)
            else:
                # what is meant here is to check if fkey combinations are unique
                are_fkeys_unique = (
                    np.unique(df[fkey_cols].to_numpy(int), axis=0).shape[0]
                    == df[fkey_cols].shape[0]
                )
                self.has_unique_fkeys[table] = (len(fkey_cols), are_fkeys_unique)

        print(
            "has unique fkey values? (number of fkeys, answer):", self.has_unique_fkeys
        )

    @staticmethod
    def listdiff(a, b):
        """
        Given two arrays of potentially duplicated elements,
        this function returns a new array containing all elements in a
        that are not in b.
        Example:
        listdiff([1, 2, 2, 3, 3], [2, 3, 3]) == array([1, 2]).
        Used to remove already used stubs when doing multi-parent matching.
        """
        # Count elements
        count_a = Counter(a)
        count_b = Counter(b)

        # Subtract counts
        diff_counter = count_a - count_b  # subtracts, but only keeps positive counts

        # Reconstruct the result as a flat list, then convert to array
        result = np.array(
            [elem for elem, count in diff_counter.items() for _ in range(count)]
        )
        return result

    def initialize_features_for_sampled_structure(self):
        self.synthetic_table_to_df = {}
        for table, df in self.real_table_to_df.items():
            # identify non-id columns
            cols_no_id = [col for col in df.columns if not col.endswith("_id")]
            synthetic_df = self.synthetic_table_to_structure[table]  # sampled structure
            # add non-id columns to sampled structure
            if self.dimension_tables is not None and table in self.dimension_tables:
                synthetic_df[cols_no_id] = deepcopy(df[cols_no_id])
            else:
                synthetic_df[cols_no_id] = np.zeros(
                    (len(synthetic_df), len(cols_no_id))
                )
            # have the same order as the real database and remove in_degree columns
            self.synthetic_table_to_df[table] = synthetic_df[df.columns]

    def save_generated_graph(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for table_name, df in self.synthetic_table_to_df.items():
            df.to_csv(os.path.join(save_dir, f"{table_name}.csv"), index=False)
        # copy dataset_meta and domain files
        dataset_meta_path = os.path.join(self.real_data_path, "dataset_meta.json")
        shutil.copy(dataset_meta_path, save_dir)
        for table_name in self.synthetic_table_to_df.keys():
            domain_path = os.path.join(self.real_data_path, f"{table_name}_domain.json")
            shutil.copy(domain_path, save_dir)
