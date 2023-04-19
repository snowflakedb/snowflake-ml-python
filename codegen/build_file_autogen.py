"""
Tool to auto-generate build files with genrules to auto-generate wrappers and integ tests for estimators.

Usage:

python3 snowflake/ml/experimental/amauser/transformer/build_file_autogen.py
"""
import os
from dataclasses import dataclass
from typing import List

import inflection
from absl import app
from sklearn_wrapper_autogen import AutogenTool


@dataclass(frozen=True)
class ModuleInfo:
    module_name: str
    exclude_list: List[str]


MODULES = [
    ModuleInfo("sklearn.linear_model", ["OrthogonalMatchingPursuitCV", "QuantileRegressor"]),
    ModuleInfo(
        "sklearn.ensemble",
        [
            "RandomTreesEmbedding",
            "StackingClassifier",
        ],
    ),
    ModuleInfo("sklearn.svm", ["OneClassSVM"]),
    ModuleInfo("sklearn.neural_network", []),
    ModuleInfo("sklearn.tree", ["BaseDecisionTree"]),  # Excluded BaseDecisionTree which is a private class.
    # TODO(snandamuri): Implement support for XGBRanker
    ModuleInfo("xgboost", ["Booster", "XGBModel", "XGBRanker"]),  # Excluded private classes and Ranker.
    ModuleInfo("sklearn.calibration", ["_SigmoidCalibration"]),  # Abstract base classes.
    ModuleInfo("sklearn.cluster", []),
    ModuleInfo("sklearn.compose", []),
    ModuleInfo("sklearn.covariance", []),
    # ModuleInfo("sklearn.cross_decomposition", []),
    ModuleInfo("sklearn.decomposition", ["MiniBatchNMF", "NMF", "SparseCoder", "LatentDirichletAllocation"]),
    ModuleInfo("sklearn.discriminant_analysis", []),
    # ModuleInfo("sklearn.feature_extraction", []),
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
    ModuleInfo("sklearn.gaussian_process", []),
    ModuleInfo("sklearn.impute", ["SimpleImputer"]),
    ModuleInfo("sklearn.isotonic", ["IsotonicRegression"]),
    ModuleInfo("sklearn.kernel_approximation", []),
    ModuleInfo("sklearn.kernel_ridge", []),
    ModuleInfo("sklearn.manifold", ["LocallyLinearEmbedding"]),
    ModuleInfo("sklearn.mixture", []),
    ModuleInfo("sklearn.model_selection", []),
    ModuleInfo("sklearn.multiclass", ["_ConstantPredictor"]),
    ModuleInfo(
        "sklearn.multioutput",
        [
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
    ModuleInfo("sklearn.naive_bayes", ["_BaseDiscreteNB", "_BaseNB"]),
    ModuleInfo("sklearn.neighbors", ["KNeighborsTransformer", "RadiusNeighborsTransformer"]),
    ModuleInfo("sklearn.semi_supervised", ["SelfTrainingClassifier"]),
    ModuleInfo(
        "lightgbm",
        [
            # Exclude ranker. TODO(snandamuri): Add support rankers.
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

SRC_OUTPUT_PATH = ""
TEST_OUTPUT_PATH = "tests/integ"


def indent(baseString: str, spaces: int = 0) -> str:
    """Prepend specified number of space charecters to the source string.

    Args:
        baseString: Source string.
        spaces: Number of space charecters to be prepended to the source string.

    Returns:
        Result of prepending #`spaces` number of space chars to the `baseString`.
    """
    return " " * spaces + baseString


def main(argv: List[str]) -> None:
    del argv  # Unused.

    # For each module
    for module in MODULES:
        module_root_dir = AutogenTool.module_root_dir(module.module_name)
        estimators_info_file_path = os.path.join(module_root_dir, "estimators_info.bzl")
        src_build_file_path = os.path.join(SRC_OUTPUT_PATH, module_root_dir, "BUILD.bazel")
        test_build_file_path = os.path.join(TEST_OUTPUT_PATH, module_root_dir, "BUILD.bazel")

        # Estimators info file:
        # Contains list of estimator class to auto generate for the module.
        os.makedirs("/".join(estimators_info_file_path.split("/")[:-1]), exist_ok=True)
        open(estimators_info_file_path, "w").write(get_estimators_info_file_content(module=module))

        # Src build file:
        # Contains genrules and py_library rules for all the estimator wrappers.
        src_build_file_content = (
            'load("//codegen:codegen_rules.bzl", "autogen_estimators", "autogen_init_file_for_module")\n'
            f'load(":estimators_info.bzl", "estimator_info_list")\n'
            'package(default_visibility = ["//visibility:public"])\n'
            f'\nautogen_init_file_for_module(module="{module.module_name}")'
            f'\nautogen_estimators(module="{module.module_name}", estimator_info_list=estimator_info_list)\n'
        )
        os.makedirs("/".join(src_build_file_path.split("/")[:-1]), exist_ok=True)
        open(src_build_file_path, "w").write(src_build_file_content)

        # Test build file:
        # Contains genrules and py_test rules for all the estimator wrappers.
        test_build_file_content = (
            'load("//codegen:codegen_rules.bzl", "autogen_tests_for_estimators")\n'
            f'load("//{module_root_dir}:estimators_info.bzl", "estimator_info_list")\n'
            'package(default_visibility = ["//visibility:public"])\n'
            "\nautogen_tests_for_estimators(\n"
            f'    module = "{module.module_name}",\n'
            f'    module_root_dir = "{module_root_dir}",\n'
            "    estimator_info_list=estimator_info_list\n"
            ")\n"
        )
        os.makedirs("/".join(test_build_file_path.split("/")[:-1]), exist_ok=True)
        open(test_build_file_path, "w").write(test_build_file_content)


def get_estimators_info_file_content(module: ModuleInfo) -> str:
    """Returns information of all the transformer and estimator classes in the given module.

    Args:
        module: Name of the module to inspect.

    Returns:
        Information of all the transformer and estimator classes in the given module.
    """
    class_list = AutogenTool.get_estimator_class_names(module_name=module.module_name)
    estimators_info_str = (
        "estimator_info_list = [\n"
        + (
            ",\n".join(
                [
                    indent(f'struct(class_name="{c}", normalized_class_name="{inflection.underscore(c)}")', 4)
                    for c in class_list
                    if c not in module.exclude_list
                ]
            )
        )
        + "\n]\n"
    )
    return estimators_info_str


if __name__ == "__main__":
    app.run(main)
