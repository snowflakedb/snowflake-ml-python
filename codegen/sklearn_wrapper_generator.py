# TODO(snandamuri): Rename this file to estimator_wrapper_generator_lib.py
import inspect
import os
import textwrap
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

import inflection
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

NP_CONSTANTS = [c for c in dir(np) if type(getattr(np, c, None)) == float or type(getattr(np, c, None)) == int]
LOAD_BREAST_CANCER = "load_breast_cancer"
LOAD_IRIS = "load_iris"
LOAD_DIABETES = "load_diabetes"


ADDITIONAL_PARAM_DESCRIPTIONS = {
    "input_cols": """
input_cols: Optional[Union[str, List[str]]]
    A string or list of strings representing column names that contain features.
    If this parameter is not specified, all columns in the input DataFrame except
    the columns specified by label_cols, sample_weight_col, and passthrough_cols
    parameters are considered input columns. Input columns can also be set after
    initialization with the `set_input_cols` method.
    """,
    "label_cols": """
label_cols: Optional[Union[str, List[str]]]
    A string or list of strings representing column names that contain labels.
    Label columns must be specified with this parameter during initialization
    or with the `set_label_cols` method before fitting.
""",
    "output_cols": """
output_cols: Optional[Union[str, List[str]]]
    A string or list of strings representing column names that will store the
    output of predict and transform operations. The length of output_cols must
    match the expected number of output columns from the specific predictor or
    transformer class used.
    If you omit this parameter, output column names are derived by adding an
    OUTPUT_ prefix to the label column names for supervised estimators, or
    OUTPUT_<IDX>for unsupervised estimators. These inferred output column names
    work for predictors, but output_cols must be set explicitly for transformers.
    In general, explicitly specifying output column names is clearer, especially
    if you don’t specify the input column names.
    To transform in place, pass the same names for input_cols and output_cols.
    be set explicitly for transformers. Output columns can also be set after
    initialization with the `set_output_cols` method.
""",
    "sample_weight_col": """
sample_weight_col: Optional[str]
    A string representing the column name containing the sample weights.
    This argument is only required when working with weighted datasets. Sample
    weight column can also be set after initialization with the
    `set_sample_weight_col` method.
""",
    "passthrough_cols": """
passthrough_cols: Optional[Union[str, List[str]]]
    A string or a list of strings indicating column names to be excluded from any
    operations (such as train, transform, or inference). These specified column(s)
    will remain untouched throughout the process. This option is helpful in scenarios
    requiring automatic input_cols inference, but need to avoid using specific
    columns, like index columns, during training or inference. Passthrough columns
    can also be set after initialization with the `set_passthrough_cols` method.
""",
    "drop_input_cols": """
drop_input_cols: Optional[bool], default=False
    If set, the response of predict(), transform() methods will not contain input columns.
""",
}

ADDITIONAL_METHOD_DESCRIPTION = """
Raises:
    TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

Args:
    dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
        Snowpark or Pandas DataFrame.
"""


class WrapperGeneratorFactory:
    """
    Reads an estimator class descriptor and generates a WrapperGenerator object which will have
    appropriate fields to fill a template string.

    Example
    -------
    template = open(self.template_path).read()
    generator = WrapperGeneratorFactory.read(<meta_class_obj>)

    print(template_string.format(skl=generator))
    """

    @staticmethod
    def _is_class_of_type(klass: type, expected_type_name: str) -> bool:
        """Checks if the given class is in the inheritance tree of the expected type.

        Args:
            klass: The class to check for inheritance.
            expected_type_name: The name of the class to check as the parent class.

        Returns:
            True if `class_name` inherits from `expected_type_name`, otherwise False.
        """
        for c in inspect.getmro(klass):
            if c.__name__ == expected_type_name:
                return True
        return False

    @staticmethod
    def _is_transformer_obj(class_object: Tuple[str, type]) -> bool:
        """Checks if the given object is a data transformer object.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from TransformerMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "TransformerMixin")

    @staticmethod
    def _is_classifier_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object can learn and predict sparse labels values (typically strings or
        bounded number of integer numerical label values).

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from ClassifierMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "ClassifierMixin")

    @staticmethod
    def _is_cluster_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object can cluster features and conduct fit_predict methods.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from ClusterMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "ClusterMixin")

    @staticmethod
    def _is_meta_estimator_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object requires an `estimator` parameter.
        bounded number of integer numerical label values).

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from MetaEstimatorMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "MetaEstimatorMixin")

    @staticmethod
    def _is_selector_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object performs feature selection given a support mask.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from SelectorMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "SelectorMixin")

    @staticmethod
    def _is_regressor_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object can learn and predict continuous numerical label values
        (typically float label values).

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from RegressorMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "RegressorMixin")

    @staticmethod
    def _is_data_module_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given class belongs to the SKLearn data module.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class belongs to `sklearn.preprocessing._data` module, otherwise False.
        """
        return class_object[1].__module__ == "sklearn.preprocessing._data"

    @staticmethod
    def _is_multioutput_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator can learn and predict multiple labels (multi-label not multi-class)
        at a time.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from MultiOutputMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "MultiOutputMixin")

    @staticmethod
    def _is_multioutput_estimator_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator is a multioutput estimator.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from _MultiOutputEstimator, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "_MultiOutputEstimator")

    @staticmethod
    def _is_k_neighbors_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator is a k-neighbors estimator.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from KNeighborsMixin, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "KNeighborsMixin")

    @staticmethod
    def _is_xgboost(module_name: str) -> bool:
        """Checks if the given module belongs to XGBoost package.

        Args:
            module_name: Name of the module which needs to be checked.

        Returns:
            True if the module belongs to XGBoost package, otherwise False.
        """
        return module_name.split(".")[0] == "xgboost"

    @staticmethod
    def _is_lightgbm(module_name: str) -> bool:
        """Checks if the given module belongs to LightGBM package.

        Args:
            module_name: Name of the module which needs to be checked.

        Returns:
            True if the module belongs to LightGBM package, otherwise False.
        """
        return module_name.split(".")[0] == "lightgbm"

    @staticmethod
    def _is_heterogeneous_ensemble_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object is ensemble of learners(voting, stacking).

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from _BaseHeterogeneousEnsemble, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "_BaseHeterogeneousEnsemble")

    @staticmethod
    def _is_stacking_ensemble_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object is staking ensemble of learners.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from _BaseStacking, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "_BaseStacking")

    @staticmethod
    def _is_voting_ensemble_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object is voting ensemble of learners.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from _BaseVoting, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "_BaseVoting")

    @staticmethod
    def _is_chain_multioutput_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object is of type chain multioutput meta-estimator.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from _BaseChain, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "_BaseChain")

    @staticmethod
    def _is_hist_gradient_boosting_regressor_obj(class_object: Tuple[str, type]) -> bool:
        """Check if the given estimator object is of type histogram-based gradient boosting regression tree.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if the class inherits from HistGradientBoostingRegressor, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "HistGradientBoostingRegressor")

    @staticmethod
    def _is_single_col_input(class_object: Tuple[str, type]) -> bool:
        """Check if given estimator object can only accept one column at a time.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if class inherits from IsotonicRegression, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "IsotonicRegression")

    @staticmethod
    def _is_positive_value_input(class_object: Tuple[str, type]) -> bool:
        """Check if given estimator object can only accept positive values input.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if class inherits from AdditiveChi2Sampler, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "AdditiveChi2Sampler")

    @staticmethod
    def _is_grid_search_cv(class_object: Tuple[str, type]) -> bool:
        """Check if given module is GridSearchCV.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if class is GridSearchCV, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "GridSearchCV")

    @staticmethod
    def _is_randomized_search_cv(class_object: Tuple[str, type]) -> bool:
        """Check if given module is RandomizedSearchCV.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if class is RandomizedSearchCV, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "RandomizedSearchCV")

    @staticmethod
    def _is_column_transformer(class_object: Tuple[str, type]) -> bool:
        """Check if given module is ColumnTransformer.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if class inherits from ColumnTransformer, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "ColumnTransformer")

    @staticmethod
    def _is_iterative_imputer(class_object: Tuple[str, type]) -> bool:
        """Check if given module is IterativeImputer.

        Args:
            class_object: Meta class object which needs to be checked.

        Returns:
            True if class inherits from IterativeImputer, otherwise False.
        """
        return WrapperGeneratorFactory._is_class_of_type(class_object[1], "IterativeImputer")

    @staticmethod
    def get_snow_ml_module_name(module_name: str) -> str:
        """Maps source estimator module name to SnowML module name.

        Args:
            module_name: Source module name.

        Returns:
            Module name in the SnowML package.
        """
        tokens = module_name.split(".")
        if tokens[0] == "sklearn":
            return "snowflake.ml.modeling." + ".".join(module_name.split(".")[1:])
        else:
            return "snowflake.ml.modeling." + module_name

    @staticmethod
    def can_generate_wrapper(class_object: Tuple[str, type]) -> bool:
        """Returns if the generator can generate wrapper and test classes for the given meta class object.

        Args:
            class_object: Meta class object of the estimator.

        Returns:
            True if generator tool can generate snowflake wrappers and integration tests for the given class.
            False otherwise.
        """
        interesting_methods = ["fit", "transform", "fit_transform", "predict", "fit_predict"]
        for tup in inspect.getmembers(class_object[1]):
            if tup[0] in interesting_methods and inspect.isfunction(tup[1]):
                return True
        return False

    @staticmethod
    def read(class_object: Tuple[str, type], module_name: str) -> "WrapperGeneratorBase":
        """
        Read a scikit-learn estimator class object and return generator object with appropriate
        template fields filled in.

        Args:
            class_object: Scikit-learn class object.
            module_name: Root module name of the estimator object.

        Returns:
            Generator object with appropriate fields to fill in wrapper and test templates.
        """
        if WrapperGeneratorFactory._is_xgboost(module_name=module_name):
            generator: WrapperGeneratorBase = XGBoostWrapperGenerator(
                module_name=module_name, class_object=class_object
            )
        elif WrapperGeneratorFactory._is_lightgbm(module_name=module_name):
            generator = LightGBMWrapperGenerator(module_name=module_name, class_object=class_object)
        else:
            generator = SklearnWrapperGenerator(module_name=module_name, class_object=class_object)

        return generator.generate()


class WrapperGeneratorBase:
    """
    Reads a class descriptor and generates the transformed and adapted
    fields to fill a template string. The member fields provided are listed below. All
    members can be referred to in the template string.

    ------------------------------------------------------------------------------------
    NAMES
    ------------------------------------------------------------------------------------

    original_class_name              INFERRED   Class name for the given scikit-learn
                                                estimator.
    module_name                      INFERRED   Name of the module that given class is
                                                is contained in.
    estimator_imports                GENERATED  Imports needed for the estimator / fit()
                                                call.
    ------------------------------------------------------------------------------------
    SIGNATURES AND ARGUMENTS
    ------------------------------------------------------------------------------------

    original_init_signature          INFERRED   Signature of the given scikit-learn
                                                class
    estimator_init_signature         GENERATED  Signature of the __init__() method of
                                                the generated estimator class.
    sklearn_init_arguments           GENERATED  Argument string to be passed to the
                                                constructor of the wrapped scikit-learn
                                                class.
    estimator_init_member_args       GENERATED  [DO NOT USE] Initializer for the members
                                                of the generated estimator.
    predict_udf_deps                 GENERATED  List of package required for predict UDF.
    fit_sproc_deps                   GENERATED  List of package required for fit sproc.

    ------------------------------------------------------------------------------------
    DOCSTRINGS (currently mostly unsupported)
    ------------------------------------------------------------------------------------

    original_class_docstring         INFERRED   Class-level docstring from the given
                                                scikit-learn class.
    estimator_class_docstring        GENERATED  Docstring describing the generated
                                                estimator class.
    transformer_class_docstring      GENERATED  Docstring describing the generated
                                                transformer class.

    original_fit_docstring           INFERRED   Docstring for the fit() method taken
                                                from the given estimator.
    estimator_fit_docstring          GENERATED  Docstring for the fit() method of the
                                                generated class.
    original_transform_docstring     INFERRED   Docstring from the given class'
                                                transform() or predict() method.
    transformer_transform_docstring  GENERATED  Docstring for the transform() or
                                                predict() method of the generated class.
    score_docstring                  GENERATED  Docstring for the score() method of the
                                                generated class.
    ------------------------------------------------------------------------------------
    TEST STRINGS
    ------------------------------------------------------------------------------------

    test_dataset_func               INFERRED    Function name to generate datasets for
                                                testing estimators.

    test_estimator_input_args       GENERATED   Input arguments string to initialize
                                                estimator.

    test_class_name                 GENERATED   Class name for integration tests.

    Caveats and limitations:
     * Scikit-learn does not specify argument types. Inferring types from default
       arguments is error-prone and disabled by default. To enable it, set
       infer_signature_types = True before calling read().
     * _reset functions are not supported. They should not be needed in split
       estimator/transformer scenarios.
    """

    def __init__(self, module_name: str, class_object: Tuple[str, type]) -> None:
        self.root_module_name = module_name
        self.module_name = module_name
        self.class_object = class_object
        self.infer_signature_types = False

        # Naming of the class.
        self.original_class_name = ""

        # The signature and argument passing the __init__ functions.
        self.original_init_signature = inspect.Signature()
        self.estimator_init_signature = ""
        self.estimator_args_transform_calls = ""
        self.sklearn_init_arguments = ""
        self.sklearn_init_args_dict = ""
        self.estimator_init_member_args = ""

        # Doc strings
        self.original_class_docstring = ""
        self.estimator_class_docstring = ""
        self.estimator_function_docstring: Dict[str, str] = {}
        self.transformer_class_docstring = ""
        self.original_fit_docstring = ""
        self.fit_docstring = ""
        self.original_transform_docstring = ""
        self.transform_docstring = ""
        self.original_predict_docstring = ""
        self.predict_docstring = ""
        self.fit_predict_docstring = ""
        self.predict_proba_docstring = ""
        self.score_docstring = ""
        self.predict_log_proba_docstring = ""
        self.decision_function_docstring = ""
        self.kneighbors_docstring = ""

        # Import strings
        self.estimator_imports = ""
        self.estimator_imports_list: List[str] = []
        self.score_sproc_imports: List[str] = []
        self.additional_import_statements = ""

        # Test strings
        self.test_dataset_func = ""
        self.test_estimator_input_args = ""
        self.test_estimator_input_args_list: List[str] = []
        self.test_class_name = ""
        self.test_estimator_imports = ""
        self.test_estimator_imports_list: List[str] = []

        # Optional function support
        self.fit_predict_cluster_function_support = False

        # Dependencies
        self.predict_udf_deps = ""
        self.fit_sproc_deps = ""

        # Native function transform
        self.supported_export_method = ""

    def _format_default_value(self, default_value: Any) -> str:
        if isinstance(default_value, str):
            return f'"{default_value}"'

        # if default values is a named numpy constant
        if (isinstance(default_value, int) or isinstance(default_value, float)) and (
            str(default_value) in NP_CONSTANTS or str(default_value).lstrip("-") in NP_CONSTANTS
        ):
            # numpy is already imported in all the wrapper classes with alias np.
            return f"{'-' if str(default_value).startswith('-') else ''}np.{str(default_value).lstrip('-')}"

        if inspect.isfunction(default_value):
            import_stmt = f"import {default_value.__module__}"
            if import_stmt not in self.estimator_imports_list:
                self.estimator_imports_list.append(import_stmt)
            return f"{default_value.__module__}.{default_value.__qualname__}"

        return str(default_value)

    def _format_default_type(self, default_value: Any) -> str:
        # TODO(amauser): add type mapping when necessary (e.g. tuple -> Tuple).
        type_name = type(default_value).__name__
        if isinstance(default_value, Iterable):
            internal_types = [type(x).__name__ for x in default_value]
            return "{}[{}]".format(type_name, ", ".join(internal_types))
        else:
            return type_name

    def _populate_flags(self) -> None:
        self._from_data_py = WrapperGeneratorFactory._is_data_module_obj(self.class_object)
        self._is_regressor = WrapperGeneratorFactory._is_regressor_obj(self.class_object)
        self._is_classifier = WrapperGeneratorFactory._is_classifier_obj(self.class_object)
        self._is_meta_estimator = WrapperGeneratorFactory._is_meta_estimator_obj(self.class_object)
        self._is_selector = WrapperGeneratorFactory._is_selector_obj(self.class_object)
        self._is_transformer = WrapperGeneratorFactory._is_transformer_obj(self.class_object)
        self._is_multioutput = WrapperGeneratorFactory._is_multioutput_obj(self.class_object)
        self._is_multioutput_estimator = WrapperGeneratorFactory._is_multioutput_estimator_obj(self.class_object)
        self._is_k_neighbors = WrapperGeneratorFactory._is_k_neighbors_obj(self.class_object)
        self._is_heterogeneous_ensemble = WrapperGeneratorFactory._is_heterogeneous_ensemble_obj(self.class_object)
        self._is_cluster = WrapperGeneratorFactory._is_cluster_obj(self.class_object)
        self._is_stacking_ensemble = WrapperGeneratorFactory._is_stacking_ensemble_obj(self.class_object)
        self._is_voting_ensemble = WrapperGeneratorFactory._is_voting_ensemble_obj(self.class_object)
        self._is_chain_multioutput = WrapperGeneratorFactory._is_chain_multioutput_obj(self.class_object)
        self._is_hist_gradient_boosting_regressor = WrapperGeneratorFactory._is_hist_gradient_boosting_regressor_obj(
            self.class_object
        )
        self._is_single_col_input = WrapperGeneratorFactory._is_single_col_input(self.class_object)
        self._is_positive_value_input = WrapperGeneratorFactory._is_positive_value_input(self.class_object)
        self._is_column_transformer = WrapperGeneratorFactory._is_column_transformer(self.class_object)
        self._is_grid_search_cv = WrapperGeneratorFactory._is_grid_search_cv(self.class_object)
        self._is_randomized_search_cv = WrapperGeneratorFactory._is_randomized_search_cv(self.class_object)
        self._is_iterative_imputer = WrapperGeneratorFactory._is_iterative_imputer(self.class_object)

    def _populate_import_statements(self) -> None:
        self.estimator_imports_list.append("import numpy")
        if self.original_class_name == "IterativeImputer":
            self.estimator_imports_list.append("from sklearn.experimental import enable_iterative_imputer")
            self.test_estimator_imports_list.append("from sklearn.experimental import enable_iterative_imputer")

    def _populate_class_doc_fields(self) -> None:
        # It's possible to use inspect.getmro(transformer[1]) to get class inheritance.
        class_docstring = inspect.getdoc(self.class_object[1]) or ""
        class_docstring = class_docstring.rsplit("Attributes\n", 1)[0]

        parameters_heading = "Parameters\n----------\n"
        class_description, param_description = (
            class_docstring.rsplit(parameters_heading, 1)
            if len(class_docstring.rsplit(parameters_heading, 1)) == 2
            else (class_docstring, "")
        )

        # Extract the first sentence of the class description
        class_description = class_description.split(".", 1)[0]
        # Add link to original source
        class_description += (
            "\nFor more details on this class, see ["
            f"{self.root_module_name}.{self.original_class_name}"
            f"]\n({self.get_doc_link()})"
        )

        # Add SnowML specific param descriptions before third party parameters.
        snowml_parameters = ""
        for d in ADDITIONAL_PARAM_DESCRIPTIONS.values():
            snowml_parameters += d
        param_description = f"{parameters_heading}{snowml_parameters}\n{param_description.strip()}"

        class_docstring = f"{class_description}\n\n{param_description}"
        class_docstring = textwrap.indent(class_docstring, "    ").strip()

        # Filter out versioned and deprecated notes from SKLearn documentation.
        paragraphs = class_docstring.split("\n\n")
        filtered_paragraphs = []
        for paragraph in paragraphs:
            if not paragraph.lstrip().startswith(".."):
                # Format colon spacing consistently for sphinx.
                paragraph = paragraph.replace(" : ", ": ")
                filtered_paragraphs.append(paragraph)

        class_docstring = "\n\n".join(filtered_paragraphs)

        self.estimator_class_docstring = class_docstring

    def _populate_function_doc_fields(self) -> None:
        _METHODS = [
            "fit",
            "fit_predict",
            "predict",
            "predict_log_proba",
            "predict_proba",
            "decision_function",
            "transform",
            "score",
            "kneighbors",
        ]
        _CLASS_FUNC = {name: func for name, func in inspect.getmembers(self.class_object[1])}
        for _each_method in _METHODS:
            if _each_method in _CLASS_FUNC.keys():
                docstring = inspect.getdoc(_CLASS_FUNC[_each_method]) or ""
                func_description, _ = (
                    docstring.rsplit("Parameters\n", 1)
                    if len(docstring.rsplit("Parameters\n", 1)) == 2
                    else (docstring, "")
                )
                func_description = func_description.split(".", 1)[0]
                func_description += (
                    "\nFor more details on this function, see ["
                    f"{self.root_module_name}.{self.original_class_name}.{_each_method}"
                    f"]\n({self.get_func_link(_each_method)})"
                )
            else:
                func_description = "Method not supported for this class."

            param_description = ADDITIONAL_METHOD_DESCRIPTION
            func_docstring = f"{func_description}\n\n{param_description}"
            func_docstring = textwrap.indent(func_docstring, "        ").strip()
            self.estimator_function_docstring[_each_method] = func_docstring

        self.fit_docstring = self.estimator_function_docstring["fit"]
        self.transform_docstring = self.estimator_function_docstring["transform"]
        self.predict_docstring = self.estimator_function_docstring["predict"]
        self.fit_predict_docstring = self.estimator_function_docstring["fit_predict"]
        self.predict_proba_docstring = self.estimator_function_docstring["predict_proba"]
        self.predict_log_proba_docstring = self.estimator_function_docstring["predict_log_proba"]
        self.decision_function_docstring = self.estimator_function_docstring["decision_function"]
        self.score_docstring = self.estimator_function_docstring["score"]
        self.kneighbors_docstring = self.estimator_function_docstring["kneighbors"]

    def _populate_class_names(self) -> None:
        self.original_class_name = self.class_object[0]
        self.test_class_name = f"{self.original_class_name}Test"

    def _populate_function_names_and_signatures(self) -> None:
        for member in inspect.getmembers(self.class_object[1]):
            if member[0] == "__init__":
                self.original_init_signature = inspect.signature(member[1])
            elif member[0] == "fit":
                original_fit_signature = inspect.signature(member[1])
                if original_fit_signature.parameters["y"].default is None:
                    # The fit does not require labels, so our label_cols argument is optional.
                    ADDITIONAL_PARAM_DESCRIPTIONS[
                        "label_cols"
                    ] = """
label_cols: Optional[Union[str, List[str]]]
    This parameter is optional and will be ignored during fit. It is present here for API consistency by convention.
                    """

        signature_lines = []
        sklearn_init_lines = []
        init_member_args = []
        has_kwargs = False
        sklearn_init_args_dict_list = []

        for k, v in self.original_init_signature.parameters.items():
            if k == "self":
                signature_lines.append("self")
                signature_lines.append("*")  # End positional arguments
                continue

            if v.kind == inspect.Parameter.VAR_KEYWORD:
                has_kwargs = True
                continue

            # Infer the type of sklearn arguments is error prone. Use at your own risk.
            if v.default != inspect.Parameter.empty:
                default_value_str = self._format_default_value(v.default)
                default_type_str = self._format_default_type(v.default)
                if self.infer_signature_types:
                    signature_lines.append(f"{v.name}: {default_type_str}={default_value_str}")
                else:
                    signature_lines.append(f"{v.name}={default_value_str}")
                sklearn_init_args_dict_list.append(f"'{v.name}':({v.name}, {default_value_str}, False)")
            else:
                if self.infer_signature_types:
                    signature_lines.append(f"{v.name}: {default_type_str}")
                else:
                    signature_lines.append(v.name)
                sklearn_init_args_dict_list.append(f"'{v.name}':({v.name}, None, True)")

        for arg in ["input_cols", "output_cols", "label_cols", "passthrough_cols"]:
            signature_lines.append(f"{arg}: Optional[Union[str, Iterable[str]]] = None")
            init_member_args.append(f"self.set_{arg}({arg})")

        signature_lines.append("drop_input_cols: Optional[bool] = False")
        init_member_args.append("self.set_drop_input_cols(drop_input_cols)")

        signature_lines.append("sample_weight_col: Optional[str] = None")
        init_member_args.append("self.set_sample_weight_col(sample_weight_col)")

        sklearn_init_lines.append("**cleaned_up_init_args")
        if has_kwargs:
            signature_lines.append("**kwargs")
            sklearn_init_lines.append("**kwargs")

        args_to_transform = ["steps", "transformers", "estimator", "estimators", "base_estimator", "final_estimator"]
        arg_transform_calls = []
        deps_gathering_calls = []
        for arg_to_transform in args_to_transform:
            if arg_to_transform in self.original_init_signature.parameters.keys():
                arg_transform_calls.append(
                    f"{arg_to_transform} = transform_snowml_obj_to_sklearn_obj({arg_to_transform})"
                )
                deps_gathering_calls.append(f"deps = deps | gather_dependencies({arg_to_transform})")

        self.estimator_init_signature = ",\n        ".join(signature_lines) + ","
        self.sklearn_init_arguments = ",\n            ".join(sklearn_init_lines) + ","
        self.sklearn_init_args_dict = "{" + ",\n            ".join(sklearn_init_args_dict_list) + ",}"
        self.estimator_init_member_args = "\n        ".join(init_member_args)
        self.estimator_args_transform_calls = "\n        ".join(arg_transform_calls)
        self.estimator_args_gathering_calls = "\n        ".join(deps_gathering_calls)

        # TODO(snandamuri): Implement type inference for classifiers.
        self.udf_datatype = "float" if self._from_data_py or self._is_regressor else ""

    def _populate_file_paths(self) -> None:
        snow_ml_module_name = WrapperGeneratorFactory.get_snow_ml_module_name(self.root_module_name)
        self.estimator_file_name = os.path.join(
            "/".join(snow_ml_module_name.split(".")),
            inflection.underscore(self.original_class_name) + ".py",
        )
        self.estimator_test_file_name = os.path.join(
            "/".join(snow_ml_module_name.split(".")),
            inflection.underscore(self.original_class_name) + "_test.py",
        )

    def _populate_integ_test_fields(self) -> None:
        snow_ml_module_name = WrapperGeneratorFactory.get_snow_ml_module_name(self.root_module_name)

        if self._is_chain_multioutput:
            self.test_dataset_func = LOAD_BREAST_CANCER
        elif self._is_regressor:
            self.test_dataset_func = LOAD_DIABETES
        else:
            self.test_dataset_func = LOAD_IRIS

        self.test_estimator_imports_list.extend(
            [
                f"from {self.root_module_name} import {self.original_class_name} as Sk{self.original_class_name}",
                f"from {snow_ml_module_name} import {self.original_class_name}  # type: ignore[attr-defined]",
                f"from sklearn.datasets import {self.test_dataset_func}",
            ]
        )

        if (
            self._is_heterogeneous_ensemble
            or "estimators" in self.original_init_signature.parameters.keys()
            or "base_estimator" in self.original_init_signature.parameters.keys()
            or "estimator" in self.original_init_signature.parameters.keys()
        ):
            if self._is_stacking_ensemble:
                self.test_estimator_imports_list.append("from sklearn.model_selection import KFold")

            if self._is_regressor or self._is_iterative_imputer:
                self.test_estimator_imports_list.append(
                    "from sklearn.linear_model import (\n"
                    "    LinearRegression as SkLinearRegression,\n"
                    "    SGDRegressor as SkSGDRegressor\n"
                    ")"
                )
            else:
                self.test_estimator_imports_list.append(
                    "from sklearn.linear_model import (\n"
                    "    LogisticRegression as SkLogisticRegression,\n"
                    "    SGDClassifier as SkSGDClassifier\n"
                    ")"
                )

        if self._is_column_transformer:
            self.test_estimator_imports_list.append("from sklearn.preprocessing import StandardScaler, RobustScaler")

        if self._is_randomized_search_cv:
            self.test_estimator_imports_list.append("from scipy.stats import uniform")

    def _construct_string_from_lists(self) -> None:
        self.estimator_imports = "\n".join(self.estimator_imports_list)
        self.test_estimator_imports = "\n".join(self.test_estimator_imports_list)
        self.test_estimator_input_args = ", ".join(self.test_estimator_input_args_list)

    def generate(self) -> "WrapperGeneratorBase":
        self.module_name = ".".join(self.class_object[1].__module__.split(".")[:-1])

        self._populate_flags()
        self._populate_class_names()
        self._populate_import_statements()
        self._populate_function_doc_fields()
        self._populate_function_names_and_signatures()
        self._populate_class_doc_fields()
        self._populate_file_paths()
        self._populate_integ_test_fields()
        return self

    @abstractmethod
    def get_doc_link(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_func_link(self, method_name: str) -> str:
        raise NotImplementedError()


class SklearnWrapperGenerator(WrapperGeneratorBase):
    def generate(self) -> "SklearnWrapperGenerator":
        # Populate all the common values
        super().generate()

        # Populate SKLearn specific values
        self.estimator_imports_list.extend(["import sklearn", f"import {self.root_module_name}"])
        self.score_sproc_imports = ["sklearn"]

        if "random_state" in self.original_init_signature.parameters.keys():
            self.test_estimator_input_args_list.append("random_state=0")

        if (
            "max_iter" in self.original_init_signature.parameters.keys()
            and not self._is_hist_gradient_boosting_regressor
        ):
            self.test_estimator_input_args_list.append("max_iter=2000")

        if self.module_name == "sklearn.decomposition" and self.original_class_name == "SparseCoder":
            # `dictionary` argument for sparse coder must have dimensions (atoms, features).
            # The iris and diabetes datasets have 4 and 10 features respectively.

            n_features_dict = {
                LOAD_IRIS: 4,
                LOAD_DIABETES: 10,
                LOAD_BREAST_CANCER: 30,
            }
            dictionary = [
                [
                    1.0,
                ]
                * n_features_dict[self.test_dataset_func]
            ]
            self.test_estimator_input_args_list.append(f"dictionary={dictionary}")

        if self._is_cluster:
            self.fit_predict_cluster_function_support = True

        if WrapperGeneratorFactory._is_class_of_type(self.class_object[1], "SelectKBest"):
            # Set the k of SelectKBest features transformer to half the number of columns in the dataset.
            self.test_estimator_input_args_list.append("k=int(len(cols)/2)")

        if "n_components" in self.original_init_signature.parameters.keys():
            if WrapperGeneratorFactory._is_class_of_type(self.class_object[1], "SpectralBiclustering"):
                # For spectral bi clustering, set number of sigular vertors to consider to number of input cols and
                # num best vector to select to half the number of input cols.
                self.test_estimator_input_args_list.append("n_components=len(cols)")
                self.test_estimator_input_args_list.append("n_best=int(len(cols)/2)")
            else:
                self.test_estimator_input_args_list.append("n_components=1")

        if self._is_heterogeneous_ensemble:
            if self._is_regressor:
                self.test_estimator_input_args_list.append(
                    'estimators=[("e1", SkLinearRegression()), ("e2", SkSGDRegressor(random_state=0))]'
                )
            else:
                self.test_estimator_input_args_list.append(
                    'estimators=[("e1", SkLogisticRegression(random_state=0, max_iter=1000)), '
                    '("e2", SkSGDClassifier(random_state=0))]'
                )

            if self._is_stacking_ensemble:
                self.test_estimator_input_args_list.extend(
                    [
                        "cv=KFold(n_splits=5, shuffle=True, random_state=0)",
                        (
                            "final_estimator="
                            + (
                                "SkLinearRegression()"
                                if self._is_regressor
                                else "SkLogisticRegression(random_state=0, max_iter=1000)"
                            )
                        ),
                    ]
                )
        elif self._is_grid_search_cv:
            self.test_estimator_input_args_list.append("estimator=SkLogisticRegression(random_state=0, solver='saga')")
            self.test_estimator_input_args_list.append('param_grid={"C": [1, 10], "penalty": ("l1", "l2")}')
        elif self._is_randomized_search_cv:
            self.test_estimator_input_args_list.append("estimator=SkLogisticRegression(random_state=0, solver='saga')")
            self.test_estimator_input_args_list.append(
                'param_distributions=dict(C=uniform(loc=0, scale=4),penalty=["l2", "l1"])'
            )
        elif (
            "base_estimator" in self.original_init_signature.parameters.keys()
            or "estimator" in self.original_init_signature.parameters.keys()
        ):
            arg_name = (
                "base_estimator" if "base_estimator" in self.original_init_signature.parameters.keys() else "estimator"
            )
            if self._is_regressor or self._is_iterative_imputer:
                self.test_estimator_input_args_list.append(f"{arg_name}=SkLinearRegression()")
            else:
                self.test_estimator_input_args_list.append(
                    f"{arg_name}=SkLogisticRegression(random_state=0, max_iter=1000)"
                )

        if self._is_column_transformer:
            self.test_estimator_input_args_list.append(
                'transformers=[("ft1", StandardScaler(), cols_half_1), ("ft2", RobustScaler(), cols_half_2)]'
            )

        if self._is_hist_gradient_boosting_regressor:
            self.test_estimator_input_args_list.extend(["min_samples_leaf=1", "max_leaf_nodes=100"])

        self.deps = (
            "f'numpy=={np.__version__}', f'scikit-learn=={sklearn.__version__}', f'cloudpickle=={cp.__version__}'"
        )
        self.supported_export_method = "to_sklearn"
        self.unsupported_export_methods = ["to_xgboost", "to_lightgbm"]
        self._construct_string_from_lists()
        return self

    def get_doc_link(self) -> str:
        return (
            f"https://scikit-learn.org/stable/modules/generated/{self.root_module_name}.{self.original_class_name}.html"
        )

    def get_func_link(self, method_name: str) -> str:
        return (
            f"https://scikit-learn.org/stable/modules/generated/{self.root_module_name}.{self.original_class_name}"
            f".html#{self.root_module_name}.{self.original_class_name}.{method_name}"
        )


class XGBoostWrapperGenerator(WrapperGeneratorBase):
    def generate(self) -> "XGBoostWrapperGenerator":
        # Populate all the common values
        super().generate()

        # Populate XGBoost specific values
        self.estimator_imports_list.append("import xgboost")
        self.test_estimator_input_args_list.extend(
            ["random_state=0", "subsample=1.0", "colsample_bynode=1.0", "n_jobs=1"]
        )
        self.score_sproc_imports = ["xgboost"]
        # TODO(snandamuri): Replace cloudpickle with joblib after latest version of joblib is added to snowflake conda.
        self.supported_export_method = "to_xgboost"
        self.unsupported_export_methods = ["to_sklearn", "to_lightgbm"]
        self.deps = "f'numpy=={np.__version__}', f'xgboost=={xgboost.__version__}', f'cloudpickle=={cp.__version__}'"
        self._construct_string_from_lists()
        return self

    def get_doc_link(self) -> str:
        return (
            "https://xgboost.readthedocs.io/en/stable/python/python_api.html#"
            f"{self.root_module_name}.{self.original_class_name}"
        )

    def get_func_link(self, method_name: str) -> str:
        return (
            "https://xgboost.readthedocs.io/en/stable/python/python_api.html#"
            f"{self.root_module_name}.{self.original_class_name}.{method_name}"
        )


class LightGBMWrapperGenerator(WrapperGeneratorBase):
    def generate(self) -> "LightGBMWrapperGenerator":
        # Populate all the common values
        super().generate()

        # Populate LightGBM specific values
        self.estimator_imports_list.append("import lightgbm")
        self.test_estimator_input_args_list.extend(["random_state=0", "n_jobs=1"])
        self.score_sproc_imports = ["lightgbm"]

        self.deps = "f'numpy=={np.__version__}', f'lightgbm=={lightgbm.__version__}', f'cloudpickle=={cp.__version__}'"
        self.supported_export_method = "to_lightgbm"
        self.unsupported_export_methods = ["to_sklearn", "to_xgboost"]
        self._construct_string_from_lists()
        return self

    def get_doc_link(self) -> str:
        return (
            "https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/"
            f"{self.root_module_name}.{self.original_class_name}.html"
        )

    def get_func_link(self, method_name: str) -> str:
        return (
            "https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/"
            f"{self.root_module_name}.{self.original_class_name}.html#"
            f"{self.root_module_name}.{self.original_class_name}.{method_name}"
        )
