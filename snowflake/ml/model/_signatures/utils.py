import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model._signatures import core


def convert_list_to_ndarray(data: List[Any]) -> npt.NDArray[Any]:
    """Create a numpy array from list or nested list. Avoid ragged list and unaligned types.

    Args:
        data: List or nested list.

    Raises:
        SnowflakeMLException: ValueError: Raised when ragged nested list or list containing non-basic type confronted.
        SnowflakeMLException: ValueError: Raised when ragged nested list or list containing non-basic type confronted.

    Returns:
        The converted numpy array.
    """
    warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
    try:
        arr = np.array(data)
    except (np.VisibleDeprecationWarning, ValueError):
        # In recent version of numpy, this warning should be raised when bad list provided.
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_DATA,
            original_exception=ValueError(
                f"Unable to construct signature: Ragged nested or Unsupported list-like data {data} confronted."
            ),
        )
    warnings.filterwarnings("default", category=np.VisibleDeprecationWarning)
    if arr.dtype == object:
        # If not raised, then a array of object would be created.
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_DATA,
            original_exception=ValueError(
                f"Unable to construct signature: Ragged nested or Unsupported list-like data {data} confronted."
            ),
        )
    return arr


def rename_features(
    features: Sequence[core.BaseFeatureSpec], feature_names: Optional[List[str]] = None
) -> Sequence[core.BaseFeatureSpec]:
    """It renames the feature in features provided optional feature names.

    Args:
        features: A sequence of feature specifications and feature group specifications.
        feature_names: A list of names to assign to features and feature groups. Defaults to None.

    Raises:
        SnowflakeMLException: ValueError: Raised when provided feature_names does not match the data shape.

    Returns:
        A sequence of feature specifications and feature group specifications being renamed if names provided.
    """
    if feature_names:
        if len(feature_names) == len(features):
            for ft, ft_name in zip(features, feature_names):
                ft._name = ft_name
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"{len(feature_names)} feature names are provided, while there are {len(features)} features."
                ),
            )
    return features


def rename_pandas_df(data: pd.DataFrame, features: Sequence[core.BaseFeatureSpec]) -> pd.DataFrame:
    """It renames pandas dataframe that has non-object column index with provided features.

    Args:
        data: A pandas dataframe to be renamed.
        features: A sequence of feature specifications and feature group specifications to rename the dataframe.

    Raises:
        SnowflakeMLException: ValueError: Raised when the data does not have the same number of features as signature.

    Returns:
        A pandas dataframe with columns renamed.
    """
    df_cols = data.columns
    if df_cols.dtype in [np.int64, np.uint64, np.float64]:
        if len(features) != len(data.columns):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Data does not have the same number of features as signature. "
                    + f"Signature requires {len(features)} features, but have {len(data.columns)} in input data."
                ),
            )
        data.columns = pd.Index([feature.name for feature in features])
    return data


def huggingface_pipeline_signature_auto_infer(task: str, params: Dict[str, Any]) -> Optional[core.ModelSignature]:
    # Text

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ConversationalPipeline
    # Needs to convert to conversation object.
    if task == "conversational":
        warnings.warn(
            (
                "Conversational pipeline is removed from transformers since 4.42.0. "
                "Support will be removed from snowflake-ml-python soon."
            ),
            category=DeprecationWarning,
            stacklevel=1,
        )

        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="user_inputs", dtype=core.DataType.STRING, shape=(-1,)),
                core.FeatureSpec(name="generated_responses", dtype=core.DataType.STRING, shape=(-1,)),
            ],
            outputs=[
                core.FeatureSpec(name="generated_responses", dtype=core.DataType.STRING, shape=(-1,)),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline
    if task == "fill-mask":
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="inputs", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="outputs", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline
    if task == "ner" or task == "token-classification":
        return core.ModelSignature(
            inputs=[core.FeatureSpec(name="inputs", dtype=core.DataType.STRING)],
            outputs=[
                core.FeatureSpec(name="outputs", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline
    if task == "question-answering":
        # If top_k and topk is not set or set to 1, then the output is a dict per input, thus we could expand.
        if params.get("top_k", 1) == 1 and params.get("topk", 1) == 1:
            return core.ModelSignature(
                inputs=[
                    core.FeatureSpec(name="question", dtype=core.DataType.STRING),
                    core.FeatureSpec(name="context", dtype=core.DataType.STRING),
                ],
                outputs=[
                    core.FeatureSpec(name="score", dtype=core.DataType.DOUBLE),
                    core.FeatureSpec(name="start", dtype=core.DataType.INT64),
                    core.FeatureSpec(name="end", dtype=core.DataType.INT64),
                    core.FeatureSpec(name="answer", dtype=core.DataType.STRING),
                ],
            )
        # Else it is a list of dict per input.
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="question", dtype=core.DataType.STRING),
                core.FeatureSpec(name="context", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="outputs", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.SummarizationPipeline
    if task == "summarization":
        if params.get("return_tensors", False):
            raise NotImplementedError(
                f"Auto deployment for HuggingFace pipeline {task} "
                "when `return_tensors` set to `True` has not been supported yet."
            )
        # Always generate a dict per input
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="documents", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="summary_text", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TableQuestionAnsweringPipeline
    if task == "table-question-answering":
        # Always generate a dict per input
        # Table is a JSON serialized string
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="query", dtype=core.DataType.STRING),
                core.FeatureSpec(name="table", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="answer", dtype=core.DataType.STRING),
                core.FeatureSpec(name="coordinates", dtype=core.DataType.INT64, shape=(-1,)),
                core.FeatureSpec(name="cells", dtype=core.DataType.STRING, shape=(-1,)),
                core.FeatureSpec(name="aggregator", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline
    if task == "text-classification" or task == "sentiment-analysis":
        # If top_k is set, return a list of dict per input
        if params.get("top_k", None) is not None:
            return core.ModelSignature(
                inputs=[
                    core.FeatureSpec(name="text", dtype=core.DataType.STRING),
                    core.FeatureSpec(name="text_pair", dtype=core.DataType.STRING),
                ],
                outputs=[
                    core.FeatureSpec(name="outputs", dtype=core.DataType.STRING),
                ],
            )
        # Else, return a dict per input
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="text", dtype=core.DataType.STRING),
                core.FeatureSpec(name="text_pair", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="label", dtype=core.DataType.STRING),
                core.FeatureSpec(name="score", dtype=core.DataType.DOUBLE),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextGenerationPipeline
    if task == "text-generation":
        if params.get("return_tensors", False):
            raise NotImplementedError(
                f"Auto deployment for HuggingFace pipeline {task} "
                "when `return_tensors` set to `True` has not been supported yet."
            )
        # Always generate a list of dict per input
        return core.ModelSignature(
            inputs=[core.FeatureSpec(name="inputs", dtype=core.DataType.STRING)],
            outputs=[
                core.FeatureSpec(name="outputs", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.Text2TextGenerationPipeline
    if task == "text2text-generation":
        if params.get("return_tensors", False):
            raise NotImplementedError(
                f"Auto deployment for HuggingFace pipeline {task} "
                "when `return_tensors` set to `True` has not been supported yet."
            )
        # Always generate a dict per input
        return core.ModelSignature(
            inputs=[core.FeatureSpec(name="inputs", dtype=core.DataType.STRING)],
            outputs=[
                core.FeatureSpec(name="generated_text", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TranslationPipeline
    if task.startswith("translation"):
        if params.get("return_tensors", False):
            raise NotImplementedError(
                f"Auto deployment for HuggingFace pipeline {task} "
                "when `return_tensors` set to `True` has not been supported yet."
            )
        # Always generate a dict per input
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="inputs", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="translation_text", dtype=core.DataType.STRING),
            ],
        )

    # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
    if task == "zero-shot-classification":
        return core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="sequences", dtype=core.DataType.STRING),
                core.FeatureSpec(name="candidate_labels", dtype=core.DataType.STRING, shape=(-1,)),
            ],
            outputs=[
                core.FeatureSpec(name="sequence", dtype=core.DataType.STRING),
                core.FeatureSpec(name="labels", dtype=core.DataType.STRING, shape=(-1,)),
                core.FeatureSpec(name="scores", dtype=core.DataType.DOUBLE, shape=(-1,)),
            ],
        )

    return None
