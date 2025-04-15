import cloudpickle as cp
import numpy as np

from snowflake.ml.modeling._internal.estimator_utils import get_module_name


class ModelSpecifications:
    """
    A dataclass to define model based specifications like required imports, and package dependencies for Sproc/Udfs.
    """

    def __init__(self, imports: list[str], pkgDependencies: list[str]) -> None:
        self.imports = imports
        self.pkgDependencies = pkgDependencies


class SKLearnModelSpecifications(ModelSpecifications):
    def __init__(self) -> None:
        import sklearn

        imports: list[str] = ["sklearn"]
        # TODO(snandamuri): Replace cloudpickle with joblib after latest version of joblib is added to snowflake conda.
        pkgDependencies = [
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cp.__version__}",
        ]

        # A change from previous implementation.
        # When reusing the Sprocs for all the fit() call in the session, the static dependencies list should include
        # all the possible dependencies required during the lifetime.

        # Include XGBoost in the dependencies if it is installed.
        try:
            import xgboost
        except ModuleNotFoundError:
            pass
        else:
            pkgDependencies.append(f"xgboost=={xgboost.__version__}")

        # Include lightgbm in the dependencies if it is installed.
        try:
            import lightgbm
        except ModuleNotFoundError:
            pass
        else:
            pkgDependencies.append(f"lightgbm=={lightgbm.__version__}")

        super().__init__(imports=imports, pkgDependencies=pkgDependencies)


class XGBoostModelSpecifications(ModelSpecifications):
    def __init__(self) -> None:
        import sklearn
        import xgboost

        imports: list[str] = ["xgboost"]
        pkgDependencies: list[str] = [
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"xgboost=={xgboost.__version__}",
            f"cloudpickle=={cp.__version__}",
        ]
        super().__init__(imports=imports, pkgDependencies=pkgDependencies)


class LightGBMModelSpecifications(ModelSpecifications):
    def __init__(self) -> None:
        import lightgbm
        import sklearn

        imports: list[str] = ["lightgbm"]
        pkgDependencies: list[str] = [
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"lightgbm=={lightgbm.__version__}",
            f"cloudpickle=={cp.__version__}",
        ]
        super().__init__(imports=imports, pkgDependencies=pkgDependencies)


class SklearnModelSelectionModelSpecifications(ModelSpecifications):
    def __init__(self) -> None:
        import sklearn
        import xgboost

        imports: list[str] = ["sklearn", "xgboost"]
        pkgDependencies: list[str] = [
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cp.__version__}",
            f"xgboost=={xgboost.__version__}",
        ]

        # Only include lightgbm in the dependencies if it is installed.
        try:
            import lightgbm
        except ModuleNotFoundError:
            pass
        else:
            imports.append("lightgbm")
            pkgDependencies.append(f"lightgbm=={lightgbm.__version__}")

        super().__init__(imports=imports, pkgDependencies=pkgDependencies)


class ModelSpecificationsBuilder:
    """
    A factory class to build ModelSpecifications object for different types of models.
    """

    @classmethod
    def build(cls, model: object) -> ModelSpecifications:
        """
        A static factory method that builds ModelSpecifications object based on the module name of native model object.

        Args:
            model: Native model object to be trained.

        Returns:
            Appropriate ModelSpecification object

        Raises:
            TypeError: Raises the exception for unsupported modules.
        """
        module_name = get_module_name(model=model)
        root_module_name = module_name.split(".")[0]
        if root_module_name == "sklearn":
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

            if isinstance(model, GridSearchCV) or isinstance(model, RandomizedSearchCV):
                return SklearnModelSelectionModelSpecifications()
            return SKLearnModelSpecifications()
        elif root_module_name == "xgboost":
            return XGBoostModelSpecifications()
        elif root_module_name == "lightgbm":
            return LightGBMModelSpecifications()
        else:
            raise TypeError(
                f"Unexpected module type: {root_module_name}." "Supported module types: sklearn, xgboost, lightgbm."
            )
