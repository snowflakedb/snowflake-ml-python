import inspect
from typing import List, Optional, Tuple

import pandas as pd


class PandasModelTrainer:
    """
    A class for training machine learning models using Pandas datasets.
    """

    def __init__(
        self,
        estimator: object,
        dataset: pd.DataFrame,
        input_cols: List[str],
        label_cols: Optional[List[str]],
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
        argspec = inspect.getfullargspec(self.estimator.fit)
        args = {"X": self.dataset[self.input_cols]}

        if self.label_cols:
            label_arg_name = "Y" if "Y" in argspec.args else "y"
            args[label_arg_name] = self.dataset[self.label_cols].squeeze()

        if self.sample_weight_col is not None and "sample_weight" in argspec.args:
            args["sample_weight"] = self.dataset[self.sample_weight_col].squeeze()

        return self.estimator.fit(**args)

    def train_fit_predict(
        self,
        pass_through_columns: List[str],
        expected_output_cols_list: List[str],
    ) -> Tuple[pd.DataFrame, object]:
        """Trains the model using specified features and target columns from the dataset.
        This API is different from fit itself because it would also provide the predict
        output.

        Args:
            pass_through_columns (List[str]): The column names that would
                display in the returned dataset.
            expected_output_cols_list (List[str]): The output columns
                name as a list. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, object]: [predicted dataset, estimator]
        """
        assert hasattr(self.estimator, "fit_predict")  # make type checker happy
        args = {"X": self.dataset[self.input_cols]}
        result = self.estimator.fit_predict(**args)
        result_df = pd.DataFrame(data=result, columns=expected_output_cols_list)
        if len(pass_through_columns) == 0:
            result_df = result_df
        else:
            result_df = pd.concat([self.dataset, result_df], axis=1)
        return (result_df, self.estimator)
