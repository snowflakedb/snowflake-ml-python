from typing import Protocol


class ModelTrainer(Protocol):
    """
    Interface for model trainer implementations.

    There are multiple flavors of training like training with pandas datasets, training with
    Snowpark datasets using sprocs, and out of core training with Snowpark datasets etc.
    """

    def train(self) -> object:
        raise NotImplementedError
