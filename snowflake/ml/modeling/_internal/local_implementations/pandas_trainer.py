import inspect
from typing import Optional

import pandas as pd

from snowflake.ml.modeling._internal.estimator_utils import (
    handle_inference_result,
    is_multi_task_estimator,
)


class PandasModelTrainer:
    """
    A class for training machine learning models using Pandas datasets.
    """

    def __init__(
        self,
        estimator: object,
        dataset: pd.DataFrame,
        input_cols: list[str],
        label_cols: Optional[list[str]],
        sample_weight_col: Optional[str],
    ) -> None:
        """
        Initializes the PandasModelTrainer with a model, a Pandas DataFrame, feature, and label column names.

        Args:
            estimator: SKLearn compatible estimator or transformer object.
            dataset: The dataset used for training the model.
            input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be used for training.
            label_cols: The name(s) of one or more columns in a DataFrame representing the target variable(s) to learn.
            sample_weight_col: The column name representing the weight of training examples.
        """
        self.estimator = estimator
        self.dataset = dataset
        self.input_cols = input_cols
        self.label_cols = label_cols
        self.sample_weight_col = sample_weight_col

    def train(self) -> object:
        """
        Trains the model using specified features and target columns from the dataset.

        Returns:
            Trained model
        """
        assert hasattr(self.estimator, "fit")  # Keep mypy happy
        params = inspect.signature(self.estimator.fit).parameters
        args = {"X": self.dataset[self.input_cols]}

        if self.label_cols:
            label_arg_name = "Y" if "Y" in params else "y"
            # For multi-task estimators, avoid squeezing to maintain 2D shape
            if is_multi_task_estimator(self.estimator):
                args[label_arg_name] = self.dataset[self.label_cols]
            else:
                args[label_arg_name] = self.dataset[self.label_cols].squeeze()

        if self.sample_weight_col is not None and "sample_weight" in params:
            args["sample_weight"] = self.dataset[self.sample_weight_col].squeeze()

        return self.estimator.fit(**args)

    def train_fit_predict(
        self,
        expected_output_cols_list: list[str],
        drop_input_cols: Optional[bool] = False,
        example_output_pd_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, object]:
        """Trains the model using specified features and target columns from the dataset.
        This API is different from fit itself because it would also provide the predict
        output.

        Args:
            expected_output_cols_list (List[str]): The output columns
                name as a list. Defaults to None.
            drop_input_cols (Optional[bool]): Boolean to determine whether to
                drop the input columns from the output dataset.
            example_output_pd_df (Optional[pd.DataFrame]): Example output dataframe
                This is not used in PandasModelTrainer. It is used in SnowparkModelTrainer.

        Returns:
            Tuple[pd.DataFrame, object]: [predicted dataset, estimator]
        """
        assert hasattr(self.estimator, "fit_predict")  # make type checker happy
        result = self.estimator.fit_predict(X=self.dataset[self.input_cols])
        result_df = pd.DataFrame(data=result, columns=expected_output_cols_list)
        if drop_input_cols:
            result_df = result_df
        else:
            # in case the output column name overlap with the input column names,
            # remove the ones in input column names
            remove_dataset_col_name_exist_in_output_col = list(
                set(self.dataset.columns) - set(expected_output_cols_list)
            )
            result_df = pd.concat([self.dataset[remove_dataset_col_name_exist_in_output_col], result_df], axis=1)
        return (result_df, self.estimator)

    def train_fit_transform(
        self,
        expected_output_cols_list: list[str],
        drop_input_cols: Optional[bool] = False,
    ) -> tuple[pd.DataFrame, object]:
        """Trains the model using specified features and target columns from the dataset.
        This API is different from fit itself because it would also provide the transform
        output.

        Args:
            expected_output_cols_list (List[str]): The output columns
                name as a list. Defaults to None.
            drop_input_cols (Optional[bool]): Boolean to determine whether to
                drop the input columns from the output dataset.

        Returns:
            Tuple[pd.DataFrame, object]: [transformed dataset, estimator]
        """
        assert hasattr(self.estimator, "fit")  # make type checker happy
        assert hasattr(self.estimator, "fit_transform")  # make type checker happy

        params = inspect.signature(self.estimator.fit).parameters
        args = {"X": self.dataset[self.input_cols]}
        if self.label_cols:
            label_arg_name = "Y" if "Y" in params else "y"
            # For multi-task estimators, avoid squeezing to maintain 2D shape
            if is_multi_task_estimator(self.estimator):
                args[label_arg_name] = self.dataset[self.label_cols]
            else:
                args[label_arg_name] = self.dataset[self.label_cols].squeeze()

        if self.sample_weight_col is not None and "sample_weight" in params:
            args["sample_weight"] = self.dataset[self.sample_weight_col].squeeze()

        inference_res = self.estimator.fit_transform(**args)

        transformed_numpy_array, output_cols = handle_inference_result(
            inference_res=inference_res, output_cols=expected_output_cols_list, inference_method="fit_transform"
        )

        result_df = pd.DataFrame(data=transformed_numpy_array, columns=output_cols)
        if drop_input_cols:
            result_df = result_df
        else:
            # in case the output column name overlap with the input column names,
            # remove the ones in input column names
            remove_dataset_col_name_exist_in_output_col = list(set(self.dataset.columns) - set(output_cols))
            result_df = pd.concat([self.dataset[remove_dataset_col_name_exist_in_output_col], result_df], axis=1)
        return (result_df, self.estimator)
