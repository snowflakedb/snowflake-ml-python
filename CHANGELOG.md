# Release History

## 1.6.3

- Model Registry (PrPr) has been removed.

### Bug Fixes

- Registry: Fix a bug that when package whose name does not follow PEP-508 is provided when logging the model,
  an unexpected normalization is happening.
- Registry: Fix `not a valid remote uri` error when logging mlflow models.
- Registry: Fix a bug that `ModelVersion.run` is called in a nested way.
- Registry: Fix an issue that leads to `log_model` failure when local package version contains parts other than
  base version.

### New Features

- Data: Improve `DataConnector.to_pandas()` performance when loading from Snowpark DataFrames.
- Model Registry: Allow users to set a model task while using `log_model`.
- Feature Store: FeatureView supports ON_CREATE or ON_SCHEDULE initialize mode.

## 1.6.2 (2024-09-04)

### Bug Fixes

- Modeling: Support XGBoost version that is larger than 2.

- Data: Fix multiple epoch iteration over `DataConnector.to_torch_datapipe()` DataPipes.
- Generic: Fix a bug that when an invalid name is provided to argument where fully qualified name is expected, it will
  be parsed wrongly. Now it raises an exception correctly.
- Model Explainability: Handle explanations for multiclass XGBoost classification models
- Model Explainability: Workarounds and better error handling for XGB>2.1.0 not working with SHAP==0.42.1

### New Features

- Data: Add top-level exports for `DataConnector` and `DataSource` to `snowflake.ml.data`.
- Data: Add native batching support via `batch_size` and `drop_last_batch` arguments to `DataConnector.to_torch_dataset()`
- Feature Store: update_feature_view() supports taking feature view object as argument.

## 1.6.1 (2024-08-12)

### Bug Fixes

- Feature Store: Support large metadata blob when generating dataset
- Feature Store: Added a hidden knob in FeatureView as kargs for setting customized
  refresh_mode
- Registry: Fix an error message in Model Version `run` when `function_name` is not mentioned and model has multiple
  target methods.
- Cortex inference: snowflake.cortex.Complete now only uses the REST API for streaming and the use_rest_api_experimental
  is no longer needed.
- Feature Store: Add a new API: FeatureView.list_columns() which list all column information.
- Data: Fix `DataFrame` ingestion with `ArrowIngestor`.

### New Features

- Enable `set_params` to set the parameters of the underlying sklearn estimator, if the snowflake-ml model has been fit.
- Data: Add `snowflake.ml.data.ingestor_utils` module with utility functions helpful for `DataIngestor` implementations.
- Data: Add new `to_torch_dataset()` connector to `DataConnector` to replace deprecated DataPipe.
- Registry: Option to `enable_explainability` set to True by default for XGBoost, LightGBM and CatBoost as PuPr feature.
- Registry: Option to `enable_explainability` when registering SHAP supported sklearn models.

## 1.6.0 (2024-07-29)

### Bug Fixes

- Modeling: `SimpleImputer` can impute integer columns with integer values.
- Registry: Fix an issue when providing a pandas Dataframe whose index is not starting from 0 as the input to
  the `ModelVersion.run`.

### New Features

- Feature Store: Add overloads to APIs accept both object and name/version. Impacted APIs include read_feature_view(),
  refresh_feature_view(), get_refresh_history(), resume_feature_view(), suspend_feature_view(), delete_feature_view().
- Feature Store: Add docstring inline examples for all public APIs.
- Feature Store: Add new utility class `ExampleHelper` to help with load source data to simplify public notebooks.
- Registry: Option to `enable_explainability` when registering XGBoost models as a pre-PuPr feature.
- Feature Store: add new API `update_entity()`.
- Registry: Option to `enable_explainability` when registering Catboost models as a pre-PuPr feature.
- Feature Store: Add new argument warehouse to FeatureView constructor to overwrite the default warehouse. Also add
  a new column 'warehouse' to the output of list_feature_views().
- Registry: Add support for logging model from a model version.
- Modeling: Distributed Hyperparameter Optimization now announce GA refresh version. The latest memory efficient version
  will not have the 10GB training limitation for dataset any more. To turn off, please run
  `
  from snowflake.ml.modeling._internal.snowpark_implementations import (
      distributed_hpo_trainer,
  )
  distributed_hpo_trainer.ENABLE_EFFICIENT_MEMORY_USAGE = False
  `
- Registry: Option to `enable_explainability` when registering LightGBM models as a pre-PuPr feature.
- Data: Add new `snowflake.ml.data` preview module which contains data reading utilities like `DataConnector`
  - `DataConnector` provides efficient connectors from Snowpark `DataFrame`
  and Snowpark ML `Dataset` to external frameworks like PyTorch, TensorFlow, and Pandas. Create `DataConnector`
  instances using the classmethod constructors `DataConnector.from_dataset()` and `DataConnector.from_dataframe()`.
- Data: Add new `DataConnector.from_sources()` classmethod constructor for constructing from `DataSource` objects.
- Data: Add new `ingestor_class` arg to `DataConnector` classmethod constructors for easier `DataIngestor` injection.
- Dataset: `DatasetReader` now subclasses new `DataConnector` class.
  - Add optional `limit` arg to `DatasetReader.to_pandas()`

### Behavior Changes

- Feature Store: change some positional parameters to keyword arguments in following APIs:
  - Entity(): desc.
  - FeatureView(): timestamp_col, refresh_freq, desc.
  - FeatureStore(): creation_mode.
  - update_entity(): desc.
  - register_feature_view(): block, overwrite.
  - list_feature_views(): entity_name, feature_view_name.
  - get_refresh_history(): verbose.
  - retrieve_feature_values(): spine_timestamp_col, exclude_columns, include_feature_view_timestamp_col.
  - generate_training_set(): save_as, spine_timestamp_col, spine_label_cols, exclude_columns,
    include_feature_view_timestamp_col.
  - generate_dataset(): version, spine_timestamp_col, spine_label_cols, exclude_columns,
    include_feature_view_timestamp_col, desc, output_type.

## 1.5.4 (2024-07-11)

### Bug Fixes

- Model Registry (PrPr): Fix 401 Unauthorized issue when deploying model to SPCS.
- Feature Store: Downgrades exceptions to warnings for few property setters in feature view. Now you can set
  desc, refresh_freq and warehouse for draft feature views.
- Modeling: Fix an issue with calling `OrdinalEncoder` with `categories` as a dictionary and a pandas DataFrame
- Modeling: Fix an issue with calling `OneHotEncoder` with `categories` as a dictionary and a pandas DataFrame

### New Features

- Registry: Allow overriding `device_map` and `device` when loading huggingface pipeline models.
- Registry: Add `set_alias` method to `ModelVersion` instance to set an alias to model version.
- Registry: Add `unset_alias` method to `ModelVersion` instance to unset an alias to model version.
- Registry: Add `partitioned_inference_api` allowing users to create partitioned inference functions in registered
  models. Enable model inference methods with table functions with vectorized process methods in registered models.
- Feature Store: add 3 more columns: refresh_freq, refresh_mode and scheduling_state to the result of
  `list_feature_views()`.
- Feature Store: `update_feature_view()` supports updating description.
- Feature Store: add new API `refresh_feature_view()`.
- Feature Store: add new API `get_refresh_history()`.
- Feature Store: Add `generate_training_set()` API for generating table-backed feature snapshots.
- Feature Store: Add `DeprecationWarning` for `generate_dataset(..., output_type="table")`.
- Feature Store: `update_feature_view()` supports updating description.
- Feature Store: add new API `refresh_feature_view()`.
- Feature Store: add new API `get_refresh_history()`.
- Model Development: OrdinalEncoder supports a list of array-likes for `categories` argument.
- Model Development: OneHotEncoder supports a list of array-likes for `categories` argument.

## 1.5.3 (06-17-2024)

### Bug Fixes

- Modeling: Fix an issue causing lineage information to be missing for
  `Pipeline`, `GridSearchCV` , `SimpleImputer`, and `RandomizedSearchCV`
- Registry: Fix an issue that leads to incorrect result when using pandas Dataframe with over 100, 000 rows as the input
  of `ModelVersion.run` method in Stored Procedure.

### New Features

- Registry: Add support for TIMESTAMP_NTZ model signature data type, allowing timestamp input and output.
- Dataset: Add `DatasetVersion.label_cols` and `DatasetVersion.exclude_cols` properties.

## 1.5.2 (06-10-2024)

### Bug Fixes

- Registry: Fix an issue that leads to unable to log model in store procedure.
- Modeling: Quick fix `import snowflake.ml.modeling.parameters.enable_anonymous_sproc` cannot be imported due to package
  dependency error.

### Behavior Changes

### New Features

## 1.5.1 (05-22-2024)

### Bug Fixes

- Dataset: Fix `snowflake.connector.errors.DataError: Query Result did not match expected number of rows` when accessing
  DatasetVersion properties when case insensitive `SHOW VERSIONS IN DATASET` check matches multiple version names.
- Dataset: Fix bug in SnowFS bulk file read when used with DuckDB
- Registry: Fixed a bug when loading old models.
- Lineage: Fix Dataset source lineage propagation through `snowpark.DataFrame` transformations

### Behavior Changes

- Feature Store: convert clear() into a private function. Also make it deletes feature views and entities only.
- Feature Store: Use NULL as default value for timestamp tag value.

### New Features

- Feature Store: Added new `snowflake.ml.feature_store.setup_feature_store()` API to assist Feature Store RBAC setup.
- Feature Store: Add `output_type` argument to `FeatureStore.generate_dataset()` to allow generating data snapshots
  as Datasets or Tables.
- Registry: `log_model`, `get_model`, `delete_model` now supports fully qualified name.
- Modeling: Supports anonymous stored procedure during fit calls so that modeling would not require sufficient
  permissions to operate on schema. Please call
  `import snowflake.ml.modeling.parameters.enable_anonymous_sproc  # noqa: F401`

## 1.5.0 (05-01-2024)

### Bug Fixes

- Registry: Fix invalid parameter 'SHOW_MODEL_DETAILS_IN_SHOW_VERSIONS_IN_MODEL' error.

### Behavior Changes

- Model Development: The behavior of `fit_transform` for all estimators is changed.
  Firstly, it will cover all the estimator that contains this function,
  secondly, the output would be the union of pandas DataFrame and snowpark DataFrame.

#### Model Registry (PrPr)

`snowflake.ml.registry.artifact` and related `snowflake.ml.model_registry.ModelRegistry` APIs have been removed.

- Removed `snowflake.ml.registry.artifact` module.
- Removed `ModelRegistry.log_artifact()`, `ModelRegistry.list_artifacts()`, `ModelRegistry.get_artifact()`
- Removed `artifacts` argument from `ModelRegistry.log_model()`

#### Dataset (PrPr)

`snowflake.ml.dataset.Dataset` has been redesigned to be backed by Snowflake Dataset entities.

- New `Dataset`s can be created with `Dataset.create()` and existing `Dataset`s may be loaded
  with `Dataset.load()`.
- `Dataset`s now maintain an immutable `selected_version` state. The `Dataset.create_version()` and
  `Dataset.load_version()` APIs return new `Dataset` objects with the requested `selected_version` state.
- Added `dataset.create_from_dataframe()` and `dataset.load_dataset()` convenience APIs as a shortcut
  to creating and loading `Dataset`s with a pre-selected version.
- `Dataset.materialized_table` and `Dataset.snapshot_table` no longer exist with `Dataset.fully_qualified_name`
  as the closest equivalent.
- `Dataset.df` no longer exists. Instead, use `DatasetReader.read.to_snowpark_dataframe()`.
- `Dataset.owner` has been moved to `Dataset.selected_version.owner`
- `Dataset.desc` has been moved to `DatasetVersion.selected_version.comment`
- `Dataset.timestamp_col`, `Dataset.label_cols`, `Dataset.feature_store_metadata`, and
  `Dataset.schema_version` have been removed.

#### Feature Store (PrPr)

- `FeatureStore.generate_dataset` argument list has been changed to match the new
`snowflake.ml.dataset.Dataset` definition

  - `materialized_table` has been removed and replaced with `name` and `version`.
  - `name` moved to first positional argument
  - `save_mode` has been removed as `merge` behavior is no longer supported. The new behavior is always `errorifexists`.

- Change feature view version type from str to `FeatureViewVersion`. It is a restricted string literal.

- Remove as_dataframe arg from FeatureStore.list_feature_views(), now always returns result as DataFrame.

- Combines few metadata tags into a new tag: SNOWML_FEATURE_VIEW_METADATA. This will make previously created feature views
not readable by new SDK.

### New Features

- Registry: Add `export` method to `ModelVersion` instance to export model files.
- Registry: Add `load` method to `ModelVersion` instance to load the underlying object from the model.
- Registry: Add `Model.rename` method to `Model` instance to rename or move a model.

#### Dataset (PrPr)

- Added Snowpark DataFrame integration using `Dataset.read.to_snowpark_dataframe()`
- Added Pandas DataFrame integration using `Dataset.read.to_pandas()`
- Added PyTorch and TensorFlow integrations using `Dataset.read.to_torch_datapipe()`
    and `Dataset.read.to_tf_dataset()` respectively.
- Added `fsspec` style file integration using `Dataset.read.files()` and `Dataset.read.filesystem()`

#### Feature Store

- use new tag_reference_internal to speed up metadata lookup.

## 1.4.1 (2024-04-18)

### New Features

- Registry: Add support for `catboost` model (`catboost.CatBoostClassifier`, `catboost.CatBoostRegressor`).
- Registry: Add support for `lightgbm` model (`lightgbm.Booster`, `lightgbm.LightGBMClassifier`, `lightgbm.LightGBMRegressor`).

### Bug Fixes

- Registry: Fix a bug that leads to relax_version option is not working.

### Behavior changes

- Feature Store: update_feature_view takes refresh_freq and warehouse as argument.

## 1.4.0 (2024-04-08)

### Bug Fixes

- Registry: Fix a bug when multiple models are being called from the same query, models other than the first one will
  have incorrect result. This fix only works for newly logged model.
- Modeling: When registering a model, only method(s) that is mentioned in `save_model` would be added to model signature
  in SnowML models.
- Modeling: Fix a bug that when n_jobs is not 1, model cannot execute methods such as
  predict, predict_log_proba, and other batch inference methods. The n_jobs would automatically
  set to 1 because vectorized udf currently doesn't support joblib parallel backend.
- Modeling: Fix a bug that batch inference methods cannot infer the datatype when the first row of data contains NULL.
- Modeling: Matches Distributed HPO output column names with the snowflake identifier.
- Modeling: Relax package versions for all Distributed HPO methods if the installed version
  is not available in the Snowflake conda channel
- Modeling: Add sklearn as required dependency for LightGBM package.

### Behavior Changes

- Registry: `apply` method is no longer by default logged when logging a xgboost model. If that is required, it could
  be specified manually when logging the model by `log_model(..., options={"target_methods": ["apply", ...]})`.
- Feature Store: register_entity returns an entity object.
- Feature Store: register_feature_view `block=true` becomes default.

### New Features

- Registry: Add support for `sentence-transformers` model (`sentence_transformers.SentenceTransformer`).
- Registry: Now version name is no longer required when logging a model. If not provided, a random human readable ID
  will be generated.

## 1.3.1 (2024-03-21)

### New Features

- FileSet: `snowflake.ml.fileset.sfcfs.SFFileSystem` can now be used in UDFs and stored procedures.

## 1.3.0 (2024-03-12)

### Bug Fixes

- Registry: Fix a bug that leads to module in `code_paths` when `log_model` cannot be correctly imported.
- Registry: Fix incorrect error message when validating input Snowpark DataFrame with array feature.
- Model Registry: Fix an issue when deploying a model to SPCS that some files do not have proper permission.
- Model Development: Relax package versions for all inference methods if the installed version
  is not available in the Snowflake conda channel

### Behavior Changes

- Registry: When running the method of a model, the value range based input validation to avoid input from overflowing
  is now optional rather than enforced, this should improve the performance and should not lead to problem for most
  kinds of model. If you want to enable this check as previous, specify `strict_input_validation=True` when
  calling `run`.
- Registry: By default `relax_version=True` when logging a model instead of using the specific local dependency versions.
  This improves dependency versioning by using versions available in Snowflake. To switch back to the previous behavior
  and use specific local dependency versions, specify `relax_version=False` when calling `log_model`.
- Model Development: The behavior of `fit_predict` for all estimators is changed.
  Firstly, it will cover all the estimator that contains this function,
  secondly, the output would be the union of pandas DataFrame and snowpark DataFrame.

### New Features

- FileSet: `snowflake.ml.fileset.sfcfs.SFFileSystem` can now be serialized with `pickle`.

## 1.2.3 (2024-02-26)

### Bug Fixes

- Registry: Now when providing Decimal Type column to a DOUBLE or FLOAT feature will not error out but auto cast with
  warnings.
- Registry: Improve the error message when specifying currently unsupported `pip_requirements` argument.
- Model Development: Fix precision_recall_fscore_support incorrect results when `average="samples"`.
- Model Registry: Fix an issue that leads to description, metrics or tags are not correctly returned in newly created
  Model Registry (PrPr) due to Snowflake BCR [2024_01](https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_01/bcr-1483)

### Behavior Changes

- Feature Store: `FeatureStore.suspend_feature_view` and `FeatureStore.resume_feature_view` doesn't mutate input feature
  view argument any more. The updated status only reflected in the returned feature view object.

### New Features

- Model Development: support `score_samples` method for all the classes, including Pipeline,
  GridSearchCV, RandomizedSearchCV, PCA, IsolationForest, ...
- Registry: Support deleting a version of a model.

## 1.2.2 (2024-02-13)

### New Features

- Model Registry: Support providing external access integrations when deploying a model to SPCS. This will help and be
  required to make sure the deploying process work as long as SPCS will by default deny all network connections. The
  following endpoints must be allowed to make deployment work: docker.com:80, docker.com:443, anaconda.com:80,
  anaconda.com:443, anaconda.org:80, anaconda.org:443, pypi.org:80, pypi.org:443. If you are using
  `snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` object, the following endpoints are required
  to be allowed: huggingface.com:80, huggingface.com:443, huggingface.co:80, huggingface.co:443.

## 1.2.1 (2024-01-25)

### New Features

- Model Development: Infers output column data type for transformers when possible.
- Registry: `relax_version` option is available in the `options` argument when logging the model.

## 1.2.0 (2024-01-11)

### Bug Fixes

- Model Registry: Fix "XGBoost version not compiled with GPU support" error when running CPU inference against open-source
  XGBoost models deployed to SPCS.
- Model Registry: Fix model deployment to SPCS on Windows machines.

### New Features

- Model Development: Introduced XGBoost external memory training feature. This feature enables training XGBoost models
  on large datasets that don't fit into memory.
- Registry: New Registry class named `snowflake.ml.registry.Registry` providing similar APIs as the old one but works
  with new MODEL object in Snowflake SQL. Also, we are providing`snowflake.ml.model.Model` and
  `snowflake.ml.model.ModelVersion` to represent a model and a specific version of a model.
- Model Development: Add support for `fit_predict` method in `AgglomerativeClustering`, `DBSCAN`, and `OPTICS` classes;
- Model Development: Add support for `fit_transform` method in `MDS`, `SpectralEmbedding` and `TSNE` class.

### Additional Notes

- Model Registry: The `snowflake.ml.registry.model_registry.ModelRegistry` has been deprecated starting from version
  1.2.0. It will stay in the Private Preview phase. For future implementations, kindly utilize
  `snowflake.ml.registry.Registry`, except when specifically required. The old model registry will be removed once all
  its primary functionalities are fully integrated into the new registry.

## 1.1.2 (2023-12-18)

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

## 1.1.1 (2023-12-05)

### Bug Fixes

- Model Registry: The `predict` target method on registered models is now compatible with unsupervised estimators.
- Model Development: Fix confusion_matrix incorrect results when the row number cannot be divided by the batch size.

### New Features

- Introduced passthrough_col param in Modeling API. This new param is helpful in scenarios
  requiring automatic input_cols inference, but need to avoid using specific
  columns, like index columns, during training or inference.

## 1.1.0 (2023-12-01)

### Bug Fixes

- Model Registry: Fix panda dataframe input not handling first row properly.
- Model Development: OrdinalEncoder and LabelEncoder output_columns do not need to be valid snowflake identifiers. They
  would previously be excluded if the normalized name did not match the name specified in output_columns.

### New Features

- Model Registry: Add support for invoking public endpoint on SPCS service, by providing a "enable_ingress" SPCS
  deployment option.
- Model Development: Add support for distributed HPO - GridSearchCV and RandomizedSearchCV execution will be
  distributed on multi-node warehouses.

## 1.0.12 (2023-11-13)

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

## 1.0.11 (2023-10-27)

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

## 1.0.10 (2023-10-13)

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
