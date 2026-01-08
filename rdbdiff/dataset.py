import os

import pandas as pd
from relbench.base import Database, Dataset, Table

from rdbdiff.utils import read_json, sort_columns


class RDBDiffDataset(Dataset):
    val_timestamp = None
    test_timestamp = None

    def __init__(self, path: str, cache_dir: str = None):
        self.path = path
        # dataset_meta has the following structure:
        # :key: relation_order -> list[list[parent, child]]
        # :key: tables -> dict[table, {'children': [...], 'parents': [...]}]
        self.dataset_meta = read_json(os.path.join(self.path, "dataset_meta.json"))
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        """
        Process the raw files into a database.
        Sorts the columns of all tables such that ID columns are first, then numerical features, then categorical features.
        """
        tables = {}
        for table_name, table_metadata in self.dataset_meta["tables"].items():
            # table_metadata has the form: {'children': [...], 'parents': [...]}
            table_df = pd.read_csv(os.path.join(self.path, f"{table_name}.csv"))
            table_domain = read_json(
                os.path.join(self.path, f"{table_name}_domain.json")
            )
            table_df = sort_columns(
                table_df, table_domain
            )  # id - numerical - categorical

            pkey_col = f"{table_name}_id"
            assert pkey_col in table_df

            fkey_col_to_pkey_table = {}
            # foreign keys are stored as parents
            for foreign_table_name in table_metadata["parents"]:
                fkey_col = f"{foreign_table_name}_id"
                assert fkey_col in table_df
                fkey_col_to_pkey_table[fkey_col] = foreign_table_name

            tables[table_name] = Table(
                df=pd.DataFrame(table_df),
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col=pkey_col,
                time_col=None,
            )

        return Database(tables)
