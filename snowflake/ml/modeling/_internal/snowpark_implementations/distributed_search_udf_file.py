"""
Description:
    This is the helper file for distributed_hpo_trainer.py to create UDTF by `register_from_file`.
Performance Benefits:
    The performance benefits come from two aspects,
    1. register_from_file can reduce duplicating loading data by only loading data once in each node
    2. register_from_file enable user to load data in global variable, whereas writing UDF in python script cannot.
Developer Tips:
    Because this script is now a string, so there's no type hinting, linting, etc. It is highly recommended
    to develop in a python script, test the type hinting, and then convert it into a string.
"""

execute_template = """
from typing import Tuple, Any, List, Dict, Set, Iterator
import os
import sys
import pandas as pd
import numpy as np
import numpy.typing as npt
import cloudpickle as cp
import io


def _load_data_into_udf() -> Tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    List[List[int]],
    List[Dict[str, Any]],
    object,
    Dict[str, Any],
    Dict[str, Any],
]:
    import pyarrow.parquet as pq

    data_files = [
        filename
        for filename in os.listdir(sys._xoptions["snowflake_import_directory"])
        if filename.startswith("dataset")
    ]
    partial_df = [
        pq.read_table(os.path.join(sys._xoptions["snowflake_import_directory"], file_name)).to_pandas()
        for file_name in data_files
    ]
    df = pd.concat(partial_df, ignore_index=True)
    constant_file_path = None
    for filename in os.listdir(sys._xoptions["snowflake_import_directory"]):
        if filename.startswith("constant"):
            constant_file_path = os.path.join(sys._xoptions["snowflake_import_directory"], f"{filename}")
    if constant_file_path is None:
        raise ValueError("UDTF cannot find the constant location, abort!")
    with open(constant_file_path, mode="rb") as constant_file_obj:
        CONSTANTS = cp.load(constant_file_obj)
    df.columns = CONSTANTS['dataset_snowpark_cols']

    # load parameter grid
    local_estimator_file_path = os.path.join(
        sys._xoptions["snowflake_import_directory"],
        f"{CONSTANTS['estimator_location']}"
    )
    with open(local_estimator_file_path, mode="rb") as local_estimator_file_obj:
        estimator_objects = cp.load(local_estimator_file_obj)
        params_to_evaluate = estimator_objects["param_grid"]

    # load indices
    local_indices_file_path = os.path.join(
        sys._xoptions["snowflake_import_directory"],
        f"{CONSTANTS['indices_location']}"
    )
    with open(local_indices_file_path, mode="rb") as local_indices_file_obj:
        indices = cp.load(local_indices_file_obj)

    # load base estimator
    local_base_estimator_file_path = os.path.join(
        sys._xoptions["snowflake_import_directory"], f"{CONSTANTS['base_estimator_location']}"
    )
    with open(local_base_estimator_file_path, mode="rb") as local_base_estimator_file_obj:
        base_estimator = cp.load(local_base_estimator_file_obj)

    # load fit_and_score_kwargs
    local_fit_and_score_kwargs_file_path = os.path.join(
        sys._xoptions["snowflake_import_directory"], f"{CONSTANTS['fit_and_score_kwargs_location']}"
    )
    with open(local_fit_and_score_kwargs_file_path, mode="rb") as local_fit_and_score_kwargs_file_obj:
        fit_and_score_kwargs = cp.load(local_fit_and_score_kwargs_file_obj)

    # Convert dataframe to numpy would save memory consumption
    # Except for Pipeline, we need to keep the dataframe for the column names
    from sklearn.pipeline import Pipeline
    if isinstance(base_estimator, Pipeline):
        return (
            df[CONSTANTS['input_cols']],
            df[CONSTANTS['label_cols']].squeeze(),
            indices,
            params_to_evaluate,
            base_estimator,
            fit_and_score_kwargs,
            CONSTANTS
        )
    return (
        df[CONSTANTS['input_cols']].to_numpy(),
        df[CONSTANTS['label_cols']].squeeze().to_numpy(),
        indices,
        params_to_evaluate,
        base_estimator,
        fit_and_score_kwargs,
        CONSTANTS
    )


global_load_data = _load_data_into_udf()


# Note Table functions (UDTFs) have a limit of 500 input arguments and 500 output columns.
class SearchCV:
    def __init__(self) -> None:
        X, y, indices, params_to_evaluate, base_estimator, fit_and_score_kwargs, CONSTANTS = global_load_data
        self.X = X
        self.y = y
        self.test_indices = indices
        self.params_to_evaluate = params_to_evaluate
        self.base_estimator = base_estimator
        self.fit_and_score_kwargs = fit_and_score_kwargs
        self.fit_score_params: List[Any] = []
        self.CONSTANTS = CONSTANTS
        self.cv_indices_set: Set[int] = set()

    def process(self, idx: int, params_idx: int, cv_idx: int) -> None:
        self.fit_score_params.extend([[idx, params_idx, cv_idx]])
        self.cv_indices_set.add(cv_idx)

    def end_partition(self) -> Iterator[Tuple[int, str]]:
        from sklearn.base import clone
        from sklearn.model_selection._validation import _fit_and_score
        from sklearn.utils.parallel import Parallel, delayed

        cached_train_test_indices = {}
        # Calculate the full index here to avoid duplicate calculation (which consumes a lot of memory)
        full_index = np.arange(self.CONSTANTS['DATA_LENGTH'])
        for i in self.cv_indices_set:
            cached_train_test_indices[i] = [
                np.setdiff1d(full_index, self.test_indices[i]),
                self.test_indices[i],
            ]

        parallel = Parallel(n_jobs=self.CONSTANTS['_N_JOBS'], pre_dispatch=self.CONSTANTS['_PRE_DISPATCH'])

        out = parallel(
            delayed(_fit_and_score)(
                clone(self.base_estimator),
                self.X,
                self.y,
                train=cached_train_test_indices[split_idx][0],
                test=cached_train_test_indices[split_idx][1],
                parameters=self.params_to_evaluate[cand_idx],
                split_progress=(split_idx, self.CONSTANTS['n_splits']),
                candidate_progress=(cand_idx, self.CONSTANTS['n_candidates']),
                **self.fit_and_score_kwargs,  # load sample weight here
            )
            for _, cand_idx, split_idx in self.fit_score_params
        )

        binary_cv_results = None
        with io.BytesIO() as f:
            cp.dump(out, f)
            f.seek(0)
            binary_cv_results = f.getvalue().hex()
        yield (
            self.fit_score_params[0][0],
            binary_cv_results,
        )

SearchCV._sf_node_singleton = True
"""
