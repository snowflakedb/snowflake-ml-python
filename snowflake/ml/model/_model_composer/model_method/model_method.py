import collections
import pathlib
from typing import List, Optional, TypedDict, Union

from typing_extensions import NotRequired

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import function_generator
from snowflake.ml.model._packager.model_meta import model_meta as model_meta_api
from snowflake.snowpark._internal import type_utils


class ModelMethodOptions(TypedDict):
    """Options when creating model method.

    case_sensitive: Specify when the name of the method should be considered as case sensitive when registered to SQL.
    function_type: One of `ModelMethodFunctionTypes` specifying function type.
    """

    case_sensitive: NotRequired[bool]
    function_type: NotRequired[str]


def get_model_method_options_from_options(
    options: type_hints.ModelSaveOption, target_method: str
) -> ModelMethodOptions:
    default_function_type = model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value
    if options.get("enable_explainability", False) and target_method.startswith("explain"):
        default_function_type = model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
    method_option = options.get("method_options", {}).get(target_method, {})
    global_function_type = options.get("function_type", default_function_type)
    function_type = method_option.get("function_type", global_function_type)
    if function_type not in [function_type.value for function_type in model_manifest_schema.ModelMethodFunctionTypes]:
        raise NotImplementedError(f"Function type {function_type} is not supported.")

    return ModelMethodOptions(
        case_sensitive=method_option.get("case_sensitive", False),
        function_type=function_type,
    )


class ModelMethod:
    """A class that is responsible to create the method information in the model manifest file and call generator to
    create the function file for the method.

    Attributes:
        model_meta: Model Metadata.
        target_method: Original target method name to call with the model.
        runtime_name: Name of the Model Runtime to run the method.
        function_generator: Function file generator.
        is_partitioned_function:  Whether the model method function is partitioned.

        options: Model Method Options.
    """

    FUNCTIONS_DIR_REL_PATH = "functions"

    def __init__(
        self,
        model_meta: model_meta_api.ModelMetadata,
        target_method: str,
        runtime_name: str,
        function_generator: function_generator.FunctionGenerator,
        is_partitioned_function: bool = False,
        options: Optional[ModelMethodOptions] = None,
    ) -> None:
        self.model_meta = model_meta
        self.target_method = target_method
        self.function_generator = function_generator
        self.is_partitioned_function = is_partitioned_function
        self.runtime_name = runtime_name
        self.options = options or {}
        try:
            self.method_name = sql_identifier.SqlIdentifier(
                target_method, case_sensitive=self.options.get("case_sensitive", False)
            )
        except ValueError as e:
            raise ValueError(
                f"Your target method {self.target_method} cannot be resolved as valid SQL identifier. "
                "Try specify `case_sensitive` as True."
            ) from e

        if self.target_method not in self.model_meta.signatures.keys():
            raise ValueError(f"Target method {self.target_method} is not available in the signatures of the model.")

        self.function_type = self.options.get(
            "function_type", model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value
        )

    @staticmethod
    def _get_method_arg_from_feature(
        feature: model_signature.BaseFeatureSpec, case_sensitive: bool = False
    ) -> model_manifest_schema.ModelMethodSignatureFieldWithName:
        assert isinstance(feature, model_signature.FeatureSpec), "FeatureGroupSpec is not supported."
        try:
            feature_name = sql_identifier.SqlIdentifier(feature.name, case_sensitive=case_sensitive)
        except ValueError as e:
            raise ValueError(
                f"Your feature {feature.name} cannot be resolved as valid SQL identifier. "
                "Try specify `case_sensitive` as True."
            ) from e
        return model_manifest_schema.ModelMethodSignatureFieldWithName(
            name=feature_name.resolved(), type=type_utils.convert_sp_to_sf_type(feature.as_snowpark_type())
        )

    def save(
        self, workspace_path: pathlib.Path, options: Optional[function_generator.FunctionGenerateOptions] = None
    ) -> model_manifest_schema.ModelMethodDict:
        (workspace_path / ModelMethod.FUNCTIONS_DIR_REL_PATH).mkdir(parents=True, exist_ok=True)
        self.function_generator.generate(
            workspace_path / ModelMethod.FUNCTIONS_DIR_REL_PATH / f"{self.target_method}.py",
            self.target_method,
            self.function_type,
            self.is_partitioned_function,
            options=options,
        )
        input_list = [
            ModelMethod._get_method_arg_from_feature(ft, case_sensitive=self.options.get("case_sensitive", False))
            for ft in self.model_meta.signatures[self.target_method].inputs
        ]
        input_name_counter = collections.Counter([input_info["name"] for input_info in input_list])
        dup_input_names = [k for k, v in input_name_counter.items() if v > 1]
        if dup_input_names:
            raise ValueError(
                f"Found duplicate input feature named resolved as {', '.join(dup_input_names)} in the method"
                f" {self.target_method} This might because you have methods with same letters but different cases. "
                "In this case, set case_sensitive as True for those methods to distinguish them."
            )

        outputs: Union[
            List[model_manifest_schema.ModelMethodSignatureField],
            List[model_manifest_schema.ModelMethodSignatureFieldWithName],
        ]
        if self.function_type == model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value:
            outputs = [
                ModelMethod._get_method_arg_from_feature(ft, case_sensitive=self.options.get("case_sensitive", False))
                for ft in self.model_meta.signatures[self.target_method].outputs
            ]
        else:
            outputs = [model_manifest_schema.ModelMethodSignatureField(type="OBJECT")]

        return model_manifest_schema.ModelFunctionMethodDict(
            name=self.method_name.resolved(),
            runtime=self.runtime_name,
            type=self.function_type,
            handler=".".join(
                [ModelMethod.FUNCTIONS_DIR_REL_PATH, self.target_method, self.function_generator.FUNCTION_NAME]
            ),
            inputs=input_list,
            outputs=outputs,
        )
