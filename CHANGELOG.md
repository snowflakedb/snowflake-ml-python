# Release History

## 1.1.2

### Bug Fixes

- Generic: Fix the issue that stack trace is hidden by telemetry unexpectedly.
- Model Development: Execute model signature inference without materializing full dataframe in memory.
- Model Registry: Fix occasional 'snowflake-ml-python library does not exist' error when deploying to SPCS.

### Behavior Changes

- Model Registry: When calling `predict` with Snowpark DataFrame, both inferred or normalized column names are accepted.
- Model Registry: When logging a Snowpark ML Modeling Model, sample input data or manually provided signature will be
  ignored since they are not necessary.

### New Features

- Model Development: SQL implementation of binary `precision_score` metric.

## 1.1.1

### Bug Fixes

- Model Registry: The `predict` target method on registered models is now compatible with unsupervised estimators.
- Model Development: Fix confusion_matrix incorrect results when the row number cannot be divided by the batch size.

### New Features

- Introduced passthrough_col param in Modeling API. This new param is helpful in scenarios
  requiring automatic input_cols inference, but need to avoid using specific
  columns, like index columns, during training or inference.

## 1.1.0

### Bug Fixes

- Model Registry: Fix panda dataframe input not handling first row properly.
- Model Development: OrdinalEncoder and LabelEncoder output_columns do not need to be valid snowflake identifiers. They
  would previously be excluded if the normalized name did not match the name specified in output_columns.

### Behavior Changes

### New Features

- Model Registry: Add support for invoking public endpoint on SPCS service, by providing a "enable_ingress" SPCS
  deployment option.
- Model Development: Add support for distributed HPO - GridSearchCV and RandomizedSearchCV execution will be
  distributed on multi-node warehouses.

## 1.0.12

### Bug Fixes

- Model Registry: Fix regression issue that container logging is not shown during model deployment to SPCS.
- Model Development: Enhance the column capacity of OrdinalEncoder.
- Model Registry: Fix unbound `batch_size` error when deploying a model other than Hugging Face Pipeline
   and LLM with GPU on SPCS.

### Behavior Changes

- Model Registry: Raise early error when deploying to SPCS with db/schema that starts with underscore.
- Model Registry: `conda-forge` channel is now automatically added to channel lists when deploying to SPCS.
- Model Registry: `relax_version` will not strip all version specifier, instead it will relax `==x.y.z` specifier to
  `>=x.y,<(x+1)`.
- Model Registry: Python with different patchlevel but the same major and minor will not result a warning when loading
  the model via Model Registry and would be considered to use when deploying to SPCS.
- Model Registry: When logging a `snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` object,
  versions of local installed libraries won't be picked as dependencies of models, instead it will pick up some pre-
  defined dependencies to improve user experience.

### New Features

- Model Registry: Enable best-effort SPCS job/service log streaming when logging level is set to INFO.

## 1.0.11

### New Features

- Model Registry: Add log_artifact() public method.
- Model Development: Add support for `kneighbors`.

### Behavior Changes

- Model Registry: Change log_model() argument from TrainingDataset to List of Artifact.
- Model Registry: Change get_training_dataset() to get_artifact().

### Bug Fixes

- Model Development: Fix support for XGBoost and LightGBM models using SKLearn Grid Search and Randomized Search model selectors.
- Model Development: DecimalType is now supported as a DataType.
- Model Development: Fix metrics compatibility with Snowpark Dataframes that use Snowflake identifiers
- Model Registry: Resolve 'delete_deployment' not deleting the SPCS service in certain cases.

## 1.0.10

### Behavior Changes

- Model Development: precision_score, recall_score, f1_score, fbeta_score, precision_recall_fscore_support,
mean_absolute_error, mean_squared_error, and mean_absolute_percentage_error metric calculations are now distributed.
- Model Registry: `deploy` will now return `Deployment` for deployment information.

### New Features

- Model Registry: When the model signature is auto-inferred, it will be printed to the log for reference.
- Model Registry: For SPCS deployment, `Deployment` details will contains `image_name`, `service_spec` and `service_function_sql`.

### Bug Fixes

- Model Development: Fix an issue that leading to UTF-8 decoding errors when using modeling modules on Windows.
- Model Development: Fix an issue that alias definitions cause `SnowparkSQLUnexpectedAliasException` in inference.
- Model Registry: Fix an issue that signature inference could be incorrect when using Snowpark DataFrame as sample input.
- Model Registry: Fix too strict data type validation when predicting. Now, for example, if you have a INT8
 type feature in the signature, if providing a INT64 dataframe but all values are within the range, it would not fail.

## 1.0.9 (2023-09-28)

### Behavior Changes

- Model Development: log_loss metric calculation is now distributed.

### Bug Fixes

- Model Registry: Fix an issue that building images fails with specific docker setup.
- Model Registry: Fix an issue that unable to embed local ML library when the library is imported by `zipimport`.
- Model Registry: Fix out-of-date doc about `platform` argument in the `deploy` function.
- Model Registry: Fix an issue that unable to deploy a GPU-trained PyTorch model to a platform where GPU is not available.

## 1.0.8 (2023-09-15)

### Bug Fixes

- Model Development: Ordinal encoder can be used with mixed input column types.
- Model Development: Fix an issue when the sklearn default value is `np.nan`.
- Model Registry: Fix an issue that incorrect docker executable is used when building images.
- Model Registry: Fix an issue that specifying `token` argument when using
`snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` with `transformers < 4.32.0` is not effective.
- Model Registry: Fix an issue that incorrect system function call is used when deploying to SPCS.
- Model Registry: Fix an issue when using a `transformers.pipeline` that does not have a `tokenizer`.
- Model Registry: Fix incorrectly-inferred image repository name during model deployment to SPCS.
- Model Registry: Fix GPU resource retention issue caused by failed or stuck previous deployments in SPCS.

## 1.0.7 (2023-09-05)

### Bug Fixes

- Model Development & Model Registry: Fix an error related to `pandas.io.json.json_normalize`.
- Allow disabling telemetry.

## 1.0.6 (2023-09-01)

### New Features

- Model Registry: add `create_if_not_exists` parameter in constructor.
- Model Registry: Added get_or_create_model_registry API.
- Model Registry: Added support for using GPU inference when deploying XGBoost (`xgboost.XGBModel` and `xgboost.Booster`
), PyTorch (`torch.nn.Module` and `torch.jit.ScriptModule`) and TensorFlow (`tensorflow.Module` and
`tensorflow.keras.Model`) models to Snowpark Container Services.
- Model Registry: When inferring model signature, `Sequence` of built-in types, `Sequence` of `numpy.ndarray`,
`Sequence` of `torch.Tensor`, `Sequence` of `tensorflow.Tensor` and `Sequence` of `tensorflow.Tensor` can be used
 instead of only `List` of them.
- Model Registry: Added `get_training_dataset` API.
- Model Development: Size of metrics result can exceed previous 8MB limit.
- Model Registry: Added support save/load/deploy HuggingFace pipeline object (`transformers.Pipeline`) and our wrapper
(`snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel`) to it. Using the wrapper to specify
configurations and the model for the pipeline will be loaded dynamically when deploying. Currently, following tasks
are supported to log without manually specifying model signatures:
  - "conversational"
  - "fill-mask"
  - "question-answering"
  - "summarization"
  - "table-question-answering"
  - "text2text-generation"
  - "text-classification" (alias "sentiment-analysis" available)
  - "text-generation"
  - "token-classification" (alias "ner" available)
  - "translation"
  - "translation_xx_to_yy"
  - "zero-shot-classification"

### Bug Fixes

- Model Development: Fixed a bug when using simple imputer with numpy >= 1.25.
- Model Development: Fixed a bug when inferring the type of label columns.

### Behavior Changes

- Model Registry: `log_model()` now return a `ModelReference` object instead of a model ID.
- Model Registry: When deploying a model with 1 `target method` only, the `target_method` argument can be omitted.
- Model Registry: When using the snowflake-ml-python with version newer than what is available in Snowflake Anaconda
Channel, `embed_local_ml_library` option will be set as `True` automatically if not.
- Model Registry: When deploying a model to Snowpark Container Services and using GPU, the default value of num_workers
will be 1.
- Model Registry: `keep_order` and `output_with_input_features` in the deploy options have been removed. Now the
behavior is controlled by the type of the input when calling `model.predict()`. If the input is a `pandas.DataFrame`,
the behavior will be the same as `keep_order=True` and `output_with_input_features=False` before. If the input is a
`snowpark.DataFrame`, the behavior will be the same as `keep_order=False` and `output_with_input_features=True` before.
- Model Registry: When logging and deploying PyTorch (`torch.nn.Module` and `torch.jit.ScriptModule`) and TensorFlow
(`tensorflow.Module` and `tensorflow.keras.Model`) models, we no longer accept models whose input is a list of tensor
and output is a list of tensors. Instead, now we accept models whose input is 1 or more tensors as positional arguments,
 and output is a tensor or a tuple of tensors. The input and output dataframe when predicting keep the same as before,
 that is every column is an array feature and contains a tensor.

## 1.0.5 (2023-08-17)

### New Features

- Model Registry: Added support save/load/deploy xgboost Booster model.
- Model Registry: Added support to get the model name and the model version from model references.

### Bug Fixes

- Model Registry: Restore the db/schema back to the session after `create_model_registry()`.
- Model Registry: Fixed an issue that the UDF name created when deploying a model is not identical to what is provided
and cannot be correctly dropped when deployment getting dropped.
- connection_params.SnowflakeLoginOptions(): Added support for `private_key_path`.

## 1.0.4 (2023-07-28)

### New Features

- Model Registry: Added support save/load/deploy Tensorflow models (`tensorflow.Module`).
- Model Registry: Added support save/load/deploy MLFlow PyFunc models (`mlflow.pyfunc.PyFuncModel`).
- Model Development: Input dataframes can now be joined against data loaded from staged files.
- Model Development: Added support for non-English languages.

### Bug Fixes

- Model Registry: Fix an issue that model dependencies are incorrectly reported as unresolvable on certain platforms.

## 1.0.3 (2023-07-14)

### Behavior Changes

- Model Registry: When predicting a model whose output is a list of NumPy ndarray, the output would not be flattened,
instead, every ndarray will act as a feature(column) in the output.

### New Features

- Model Registry: Added support save/load/deploy PyTorch models (`torch.nn.Module` and `torch.jit.ScriptModule`).

### Bug Fixes

- Model Registry: Fix an issue that when database or schema name provided to `create_model_registry` contains special
characters, the model registry cannot be created.
- Model Registry: Fix an issue that `get_model_description` returns with additional quotes.
- Model Registry: Fix incorrect error message when attempting to remove a unset tag of a model.
- Model Registry: Fix a typo in the default deployment table name.
- Model Registry: Snowpark dataframe for sample input or input for `predict` method that contains a column with
Snowflake `NUMBER(precision, scale)` data type where `scale = 0` will not lead to error, and will now correctly
recognized as `INT64` data type in model signature.
- Model Registry: Fix an issue that prevent model logged in the system whose default encoding is not UTF-8 compatible
from deploying.
- Model Registry: Added earlier and better error message when any file name in the model or the file name of model
itself contains characters that are unable to be encoded using ASCII. It is currently not supported to deploy such a
model.

## 1.0.2 (2023-06-22)

### Behavior Changes

- Model Registry: Prohibit non-snowflake-native models from being logged.
- Model Registry: `_use_local_snowml` parameter in options of `deploy()` has been removed.
- Model Registry: A default `False` `embed_local_ml_library` parameter has been added to the options of `log_model()`.
With this set to `False` (default), the version of the local snowflake-ml-python library will be recorded and used when
deploying the model. With this set to `True`, local snowflake-ml-python library will be embedded into the logged model,
and will be used when you load or deploy the model.

### New Features

- Model Registry: A new optional argument named `code_paths` has been added to the arguments of `log_model()` for users
to specify additional code paths to be imported when loading and deploying the model.
- Model Registry: A new optional argument named `options` has been added to the arguments of `log_model()` to specify
any additional options when saving the model.
- Model Development: Added metrics:
  - d2_absolute_error_score
  - d2_pinball_score
  - explained_variance_score
  - mean_absolute_error
  - mean_absolute_percentage_error
  - mean_squared_error

### Bug Fixes

- Model Development: `accuracy_score()` now works when given label column names are lists of a single value.

## 1.0.1 (2023-06-16)

### Behavior Changes

- Model Development: Changed Metrics APIs to imitate sklearn metrics modules:
  - `accuracy_score()`, `confusion_matrix()`, `precision_recall_fscore_support()`, `precision_score()` methods move from
  respective modules to `metrics.classification`.
- Model Registry: The default table/stage created by the Registry now uses "_SYSTEM_" as a prefix.
- Model Registry: `get_model_history()` method as been enhanced to include the history of model deployment.

### New Features

- Model Registry: A default `False` flag named `replace_udf` has been added to the options of `deploy()`. Setting this
to `True` will allow overwrite existing UDF with the same name when deploying.
- Model Development: Added metrics:
  - f1_score
  - fbeta_score
  - recall_score
  - roc_auc_score
  - roc_curve
  - log_loss
  - precision_recall_curve
- Model Registry: A new argument named `permanent` has been added to the argument of `deploy()`. Setting this to `True`
allows the creation of a permanent deployment without needing to specify the UDF location.
- Model Registry: A new method `list_deployments()` has been added to enumerate all permanent deployments originating
from a specific model.
- Model Registry: A new method `get_deployment()` has been added to fetch a deployment by its deployment name.
- Model Registry: A new method `delete_deployment()` has been added to remove an existing permanent deployment.

## 1.0.0 (2023-06-09)

### Behavior Changes

- Model Registry: `predict()` method moves from Registry to ModelReference.
- Model Registry: `_snowml_wheel_path` parameter in options of `deploy()`, is replaced with `_use_local_snowml` with
default value of `False`. Setting this to `True` will have the same effect of uploading local SnowML code when executing
model in the warehouse.
- Model Registry: Removed `id` field from `ModelReference` constructor.
- Model Development: Preprocessing and Metrics move to the modeling package: `snowflake.ml.modeling.preprocessing` and
`snowflake.ml.modeling.metrics`.
- Model Development: `get_sklearn_object()` method is renamed to `to_sklearn()`, `to_xgboost()`, and `to_lightgbm()` for
respective native models.

### New Features

- Added PolynomialFeatures transformer to the snowflake.ml.modeling.preprocessing module.
- Added metrics:
  - accuracy_score
  - confusion_matrix
  - precision_recall_fscore_support
  - precision_score

### Bug Fixes

- Model Registry: Model version can now be any string (not required to be a valid identifier)
- Model Deployment: `deploy()` & `predict()` methods now correctly escapes identifiers

## 0.3.2 (2023-05-23)

### Behavior Changes

- Use cloudpickle to serialize and deserialize models throughout the codebase and removed dependency on joblib.

### New Features

- Model Deployment: Added support for snowflake.ml models.

## 0.3.1 (2023-05-18)

### Behavior Changes

- Standardized registry API with following
  - Create & open registry taking same set of arguments
  - Create & Open can choose schema to use
  - Set_tag, set_metric, etc now explicitly calls out arg name as metric_name, tag_name, metric_name, etc.

### New Features

- Changes to support python 3.9, 3.10
- Added kBinsDiscretizer
- Support for deployment of XGBoost models & int8 types of data

## 0.3.0 (2023-05-11)

### Behavior Changes

- Big Model Registry Refresh
  - Fixed API discrepancies between register_model & log_model.
  - Model can be referred by Name + Version (no opaque internal id is required)

### New Features

- Model Registry: Added support save/load/deploy SKL & XGB Models

## 0.2.3 (2023-04-27)

### Bug Fixes

- Allow using OneHotEncoder along with sklearn style estimators in a pipeline.

### New Features

- Model Registry: Added support for delete_model. Use delete_artifact = False to not delete the underlying model data
but just unregister.

## 0.2.2 (2023-04-11)

### New Features

- Initial version of snowflake-ml modeling package.
  - Provide support for training most of scikit-learn and xgboost estimators and transformers.

### Bug Fixes

- Minor fixes in preprocessing package.

## 0.2.1 (2023-03-23)

### New Features

- New in Preprocessing:
  - SimpleImputer
  - Covariance Matrix
- Optimization of Ordinal Encoder client computations.

### Bug Fixes

- Minor fixes in OneHotEncoder.

## 0.2.0 (2023-02-27)

### New Features

- Model Registry
- PyTorch & Tensorflow connector file generic FileSet API
- New to Preprocessing:
  - Binarizer
  - Normalizer
  - Pearson correlation Matrix
- Optimization in Ordinal Encoder to cache vocabulary in temp tables.

## 0.1.3 (2023-02-02)

### New Features

- Initial version of transformers including:
  - Label Encoder
  - Max Abs Scaler
  - Min Max Scaler
  - One Hot Encoder
  - Ordinal Encoder
  - Robust Scaler
  - Standard Scaler
