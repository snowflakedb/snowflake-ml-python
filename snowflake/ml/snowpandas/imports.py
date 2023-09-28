import dataclasses
import importlib
import inspect
from typing import Dict, List

TYPE_CLASS = "CLASS"
TYPE_FUNCTION = "FUNCTION"


@dataclasses.dataclass(frozen=True)
class ModuleInfo:
    module_name: str
    exclude_list: List[str] = dataclasses.field(default_factory=list)
    include_list: List[str] = dataclasses.field(default_factory=list)
    has_functions: bool = False


MODULES = [
    ModuleInfo("sklearn.linear_model", ["OrthogonalMatchingPursuitCV", "QuantileRegressor"]),
    ModuleInfo("sklearn.ensemble", ["RandomTreesEmbedding", "StackingClassifier"]),
    ModuleInfo("sklearn.svm", ["OneClassSVM"]),
    ModuleInfo("sklearn.neural_network"),
    ModuleInfo("sklearn.pipeline"),
    ModuleInfo("sklearn.tree", ["BaseDecisionTree"]),  # Excluded BaseDecisionTree which is a private class.
    ModuleInfo("sklearn.calibration", ["_SigmoidCalibration"]),  # Abstract base classes.
    ModuleInfo("sklearn.cluster"),
    ModuleInfo("sklearn.compose"),
    ModuleInfo("sklearn.covariance"),
    # ModuleInfo("sklearn.cross_decomposition"),
    ModuleInfo("sklearn.decomposition", ["MiniBatchNMF", "NMF", "SparseCoder", "LatentDirichletAllocation"]),
    ModuleInfo("sklearn.discriminant_analysis"),
    # ModuleInfo("sklearn.feature_extraction"),
    ModuleInfo(
        "sklearn.feature_selection",
        [
            # Abstract base classes and private classes.
            "SelectorMixin",
            # Classes to be supported in future.
            "RFE",
            "RFECV",
            "SelectFromModel",
        ],
    ),
    ModuleInfo("sklearn.gaussian_process"),
    ModuleInfo("sklearn.impute"),
    ModuleInfo("sklearn.isotonic", ["IsotonicRegression"]),
    ModuleInfo("sklearn.kernel_approximation"),
    ModuleInfo("sklearn.kernel_ridge"),
    ModuleInfo("sklearn.manifold", ["LocallyLinearEmbedding"]),
    ModuleInfo("sklearn.mixture"),
    ModuleInfo("sklearn.model_selection"),
    ModuleInfo("sklearn.multiclass", ["_ConstantPredictor"]),
    ModuleInfo(
        "sklearn.multioutput",
        [
            # Abstract base classes and private classes.
            "_BaseChain",
            "_MultiOutputEstimator",
            # Classes to be supported in future.
            "ClassifierChain",
            # "MultiOutputClassifier",
            # "MultiOutputRegressor",
            "RegressorChain",
        ],
    ),
    ModuleInfo("sklearn.naive_bayes", ["_BaseDiscreteNB", "_BaseNB"]),
    ModuleInfo("sklearn.neighbors", ["KNeighborsTransformer", "RadiusNeighborsTransformer"]),
    ModuleInfo("sklearn.preprocessing"),
    ModuleInfo("sklearn.semi_supervised", ["SelfTrainingClassifier"]),
    ModuleInfo("xgboost", ["Booster", "XGBModel", "XGBRanker"]),  # Excluded private classes and Ranker.
    ModuleInfo(
        "lightgbm",
        [
            # Exclude ranker.
            "LGBMRanker",
            # Exclude abstract classes, dask related classes.
            "LGBMModel",
            "Booster",
            "DaskLGBMClassifier",
            "DaskLGBMRanker",
            "DaskLGBMRegressor",
        ],
    ),
]


def import_modules() -> Dict[str, Dict[str, List[str]]]:
    modules: Dict[str, Dict[str, List[str]]] = {}
    for module_info in MODULES:
        try:
            module = importlib.import_module(module_info.module_name)
            class_names = []
            function_names = []
            for estimator in inspect.getmembers(module):
                if (
                    inspect.isclass(estimator[1])
                    # Not an imported class
                    and estimator[1].__module__.startswith(module_info.module_name)
                ):
                    class_names.append(estimator[0])

                if module_info.has_functions:
                    if (
                        inspect.isfunction(estimator[1])
                        # Not an imported class
                        and estimator[1].__module__.startswith(module_info.module_name)
                    ):
                        function_names.append(estimator[0])

            modules[module_info.module_name] = {}
            modules[module_info.module_name][TYPE_CLASS] = [
                v
                for v in class_names
                if (
                    not v.startswith("_")
                    and (
                        v not in module_info.exclude_list
                        if len(module_info.exclude_list) > 0
                        else v in module_info.include_list
                        if len(module_info.include_list) > 0
                        else True
                    )
                )
            ]

            modules[module_info.module_name][TYPE_FUNCTION] = [
                v
                for v in function_names
                if (
                    not v.startswith("_")
                    and (
                        v not in module_info.exclude_list
                        if len(module_info.exclude_list) > 0
                        else v in module_info.include_list
                        if len(module_info.include_list) > 0
                        else True
                    )
                )
            ]
        except ImportError:
            pass
    return modules
