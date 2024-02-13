import inspect
from typing import List, Optional

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
