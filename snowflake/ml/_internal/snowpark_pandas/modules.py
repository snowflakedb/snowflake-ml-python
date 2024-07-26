import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class ModuleInfo:
    module_name: str
    exclude_list: List[str] = dataclasses.field(default_factory=list)
    include_list: List[str] = dataclasses.field(default_factory=list)
    native_list: List[str] = dataclasses.field(default_factory=list)
    has_functions: bool = False


# TODO (xjiang): put MODULES into a constant file
MODULES = [
    ModuleInfo(
        "sklearn.linear_model",
        exclude_list=[
            "OrthogonalMatchingPursuitCV",
            "QuantileRegressor",
            "GammaRegressor",
            "PoissonRegressor",
            "TweedieRegressor",
        ],
    ),
    ModuleInfo("sklearn.ensemble", exclude_list=["RandomTreesEmbedding", "StackingClassifier"]),
    ModuleInfo("sklearn.svm", exclude_list=["OneClassSVM"]),
    ModuleInfo("sklearn.neural_network"),
    ModuleInfo("sklearn.pipeline"),
    ModuleInfo(
        "sklearn.tree", exclude_list=["BaseDecisionTree"]
    ),  # Excluded BaseDecisionTree which is a private class.
    ModuleInfo("sklearn.calibration", exclude_list=["_SigmoidCalibration"]),  # Abstract base classes.
    ModuleInfo(
        "sklearn.cluster", exclude_list=["Birch", "AffinityPropagation", "BisectingKMeans", "KMeans", "MiniBatchKMeans"]
    ),
    ModuleInfo("sklearn.compose"),
    ModuleInfo("sklearn.covariance"),
    ModuleInfo("sklearn.cross_decomposition", exclude_list=["PLSSVD"]),
    ModuleInfo(
        "sklearn.decomposition",
        exclude_list=["MiniBatchNMF", "NMF", "LatentDirichletAllocation"],
    ),
    ModuleInfo("sklearn.discriminant_analysis"),
    # ModuleInfo("sklearn.feature_extraction"),
    ModuleInfo(
        "sklearn.feature_selection",
        exclude_list=[
            # Abstract base classes and private classes.
            "SelectorMixin",
            # Classes to be supported in future.
            "RFE",
            "RFECV",
            "SelectFromModel",
        ],
    ),
    ModuleInfo("sklearn.gaussian_process"),
    ModuleInfo("sklearn.impute", native_list=["SimpleImputer"]),
    ModuleInfo("sklearn.isotonic", exclude_list=["IsotonicRegression"]),
    ModuleInfo("sklearn.kernel_approximation", exclude_list=["Nystroem"]),
    ModuleInfo("sklearn.kernel_ridge"),
    ModuleInfo("sklearn.manifold", exclude_list=["LocallyLinearEmbedding"]),
    ModuleInfo("sklearn.mixture"),
    ModuleInfo("sklearn.model_selection"),
    ModuleInfo("sklearn.multiclass", exclude_list=["_ConstantPredictor"]),
    ModuleInfo(
        "sklearn.multioutput",
        exclude_list=[
            # Abstract base classes and private classes.
            "_BaseChain",
            "_MultiOutputEstimator",
            # Classes to be supported in future.
            "ClassifierChain",
            "MultiOutputClassifier",
            "MultiOutputRegressor",
            "RegressorChain",
        ],
    ),
    ModuleInfo("sklearn.naive_bayes", exclude_list=["_BaseDiscreteNB", "_BaseNB", "CategoricalNB"]),
    ModuleInfo(
        "sklearn.preprocessing",
        native_list=[
            "Binarizer",
            "KBinsDiscretizer",
            "LabelEncoder",
            "MaxAbsScaler",
            "MinMaxScaler",
            "Normalizer",
            "OneHotEncoder",
            "OrdinalEncoder",
            "RobustScaler",
            "StandardScaler",
        ],
    ),
    ModuleInfo(
        "sklearn.neighbors",
        exclude_list=["KNeighborsTransformer", "RadiusNeighborsTransformer", "NeighborhoodComponentsAnalysis"],
    ),
    ModuleInfo("sklearn.semi_supervised", exclude_list=["SelfTrainingClassifier"]),
    ModuleInfo("xgboost", exclude_list=["Booster", "XGBModel", "XGBRanker"]),  # Excluded private classes and Ranker.
    ModuleInfo(
        "lightgbm",
        exclude_list=[
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


MODULE_MAPPING = {
    "sklearn.impute": "snowflake.ml.modeling.impute",
    "sklearn.preprocessing": "snowflake.ml.modeling.preprocessing",
}
