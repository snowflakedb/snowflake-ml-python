# Release History

## 1.17.0

### Bug Fixes

* ML Job: Added support for retrieving details of deleted jobs, including status, compute pool, and target instances.

### Behavior Changes

### New Features

* Support xgboost 3.x.
* ML Job: Overhauled the `MLJob.result()` API with broader cross-version
  compatibility and support for additional data types, namely:
  * Pandas DataFrames
  * PyArrow Tables
  * NumPy arrays
  * NOTE: Requires `snowflake-ml-python>=1.17.0` to be installed inside remote container environment.
* ML Job: Enabled job submission v2 by default
  * Jobs submitted using v2 will automatically use the latest Container Runtime image
  * v1 behavior can be restored by setting environment variable `MLRS_USE_SUBMIT_JOB_V2` to `false`

## 1.16.0

### Bug Fixes

* Registry: Remove redundant pip dependency warnings when `artifact_repository_map` is provided for warehouse model deployments.

### Behavior Changes

### New Features

* Support scikit-learn < 1.8.
* ML Job: Added support for configuring the runtime image via `runtime_environment`
  (image tag or full image URL) at submission time.
  Examples:
  * @remote(compute_pool, stage_name = 'payload_stage', runtime_environment = '1.8.0')
  * submit_file('/path/to/repo/test.py', compute_pool, stage_name = 'payload_stage', runtime_environment = '/mydb/myschema/myrepo/myimage:latest')
* Registry: Ability to mark model methods as `Volatility.VOLATILE` or `Volatility.IMMUTABLE`.

```python
from snowflake.ml.model.volatility import Volatility

options = {
    "embed_local_ml_library": True,
    "relax_version": True,
    "save_location": "/path/to/my/directory",
    "function_type": "TABLE_FUNCTION",
    "volatility": Volatility.IMMUTABLE,
    "method_options": {
        "predict": {
            "case_sensitive": False,
            "max_batch_size": 100,
            "function_type": "TABLE_FUNCTION",
            "volatility": Volatility.VOLATILE,
        },
    },
}

```

## 1.15.0 (09-29-2025)

### Bug Fixes

### Behavior Changes

* Registry: Dropping support for deprecated `conversational` task type for Huggingface models.
  To read more <https://github.com/huggingface/transformers/pull/31165>

### New Features

## 1.14.0 (09-18-2025)

### Bug Fixes

### Behavior Changes

### New Features

* ML Job: The `additional_payloads` argument is now **deprecated** in favor of `imports`.

## 1.13.0

### Bug Fixes

### Behavior Changes

### New Features

* Registry: Log a HuggingFace model without having to load the model in memory using
 the `huggingface_pipeline.HuggingFacePipelineModel`. Requires `huggingface_hub` package to installed.
 To disable downloading HuggingFace repository, provide `download_snapshot=False` while creating the
 `huggingface_pipeline.HuggingFacePipelineModel` object.
* Registry: Added support for XGBoost models to use `enable_categorical=True` with pandas DataFrame
* Registry: Added support to display privatelink inference endpoint in ModelVersion list services.

## 1.12.0

### Bug Fixes

* Registry: Fixed an issue where the string representation of dictionary-type output columns was being incorrectly
  created during structured output deserialization. Now, the original data type is properly preserved.
* Registry: Fixed the inference server performance issue for wide (500+ features) and JSON inputs.

### Behavior Changes

### New Features

* Registry: Add OpenAI chat completion compatible signature option for `text-generation` models.

```python
from snowflake.ml.model import openai_signatures
import pandas as pd

mv = snowflake_registry.log_model(
    model=generator,
    model_name=...,
    ...,
    signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
)

# create a pd.DataFrame with openai.client.chat.completions arguments like below:
x_df = pd.DataFrame.from_records(
    [
        {
            "messages": [
                {"role": "system", "content": "Complete the sentence."},
                {
                    "role": "user",
                    "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",
                },
            ],
            "max_completion_tokens": 250,
            "temperature": 0.9,
            "stop": None,
            "n": 3,
            "stream": False,
            "top_p": 1.0,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }
    ],
)

# OpenAI Chat Completion compatible output
output_df = mv.run(X=x_df)
```

* Model Monitoring: Added support for segment columns to enable filtered analysis.
  * Added `segment_columns` parameter to `ModelMonitorSourceConfig` to specify columns for segmenting monitoring data
  * Segment columns must be of STRING type and exist in the source table
  * Added methods to dynamically manage segments:
    * `add_segment_column()`: Add a new segment column to an existing monitor
    * `drop_segment_column()`: Remove a segment column from an existing monitor
* Experiment Tracking (PrPr): Support for logging artifacts (files and directories) with `log_artifact`
* Experiment Tracking (PrPr): Support for listing artifacts in a run with `list_artifacts`
* Experiment Tracking (PrPr): Support for downloading artifacts in a run with `download_artifacts`

## 1.11.0 (08-12-2025)

### Bug Fixes

* ML Job: Fix `Error: Unable to retrieve head IP address` if not all instances start within the timeout.
* ML Job: Fix `TypeError: SnowflakeCursor.execute() got an unexpected keyword argument '_force_qmark_paramstyle'`
  when running inside Stored Procedures.

### Behavior Changes

### New Features

* `ModelVersion.create_service()`: Made `image_repo` argument optional. By
  default it will use a default image repo, which is
  being rolled out in server version 9.22+.
* Experiment Tracking (PrPr): Automatically log the model, metrics, and parameters while training Keras models with
  `snowflake.ml.experiment.callback.keras.SnowflakeKerasCallback`.

## 1.10.0

### Behavior Changes

* Experiment Tracking (PrPr): The import paths for the auto-logging callbacks have changed to
  `snowflake.ml.experiment.callback.xgboost.SnowflakeXgboostCallback` and
  `snowflake.ml.experiment.callback.lightgbm.SnowflakeLightgbmCallback`.

### New Features

* Registry: add progress bars for `ModelVersion.create_service` and `ModelVersion.log_model`.
* ModelRegistry: Logs emitted during `ModelVersion.create_service` will be written to a file. The file location
  will be shown in the console.

## 1.9.2

### Bug Fixes

* DataConnector: Fix `self._session` related errors inside Container Runtime.
* Registry: Fix a bug when trying to pass `None` to array (`pd.dtype('O')`) in signature and pandas data handler.

### New Features

* Experiment Tracking (PrPr): Automatically log the model, metrics, and parameters while training
  XGBoost and LightGBM models.

```python
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.experiment.callback import SnowflakeXgboostCallback, SnowflakeLightgbmCallback

exp = ExperimentTracking(session=sp_session, database_name="ML", schema_name="PUBLIC")

exp.set_experiment("MY_EXPERIMENT")

# XGBoost
callback = SnowflakeXgboostCallback(
  exp, log_model=True, log_metrics=True, log_params=True, model_name="model_name", model_signature=sig
)
model = XGBClassifier(callbacks=[callback])
with exp.start_run():
  model.fit(X, y, eval_set=[(X_test, y_test)])

# LightGBM
callback = SnowflakeLightgbmCallback(
  exp, log_model=True, log_metrics=True, log_params=True, model_name="model_name", model_signature=sig
)
model = LGBMClassifier()
with exp.start_run():
  model.fit(X, y, eval_set=[(X_test, y_test)], callbacks=[callback])
```

## 1.9.1 (07-18-2025)

### Bug Fixes

* Registry: Fix a bug when trying to set the PAD token the HuggingFace `text-generation` model had multiple EOS tokens.
  The handler picks the first EOS token as PAD token now.

### New Features

* DataConnector: DataConnector objects can now be pickled
* Dataset: Dataset objects can now be pickled
* Registry (PrPr): Introducing `create_service` function in `snowflake/ml/model/models/huggingface_pipeline.py`
  which creates a service to log a HF model and upon successful logging, an inference service is created.

```python
from snowflake.ml.model.models import huggingface_pipeline

hf_model_ref = huggingface_pipeline.HuggingFacePipelineModel(
  model="gpt2",
  task="text-generation", # Optional
)


hf_model_ref.create_service(
    session=session,
    service_name="test_service",
    service_compute_pool="test_compute_pool",
    image_repo="test_repo",
    ...
)
```

* Experiment Tracking (PrPr): New module for managing and tracking ML experiments in Snowflake.

```python
from snowflake.ml.experiment import ExperimentTracking

exp = ExperimentTracking(session=sp_session, database_name="ML", schema_name="PUBLIC")

exp.set_experiment("MY_EXPERIMENT")

with exp.start_run():
  exp.log_param("batch_size", 32)
  exp.log_metrics("accuracy", 0.98, step=10)
  exp.log_model(my_model, model_name="MY_MODEL")
```

* Registry: Added support for wide input (500+ features) for inference done using SPCS

## 1.9.0

### Bug Fixes

* Registry: Fixed bug causing snowpark to pandas dataframe conversion to fail when `QUOTED_IDENTIFIERS_IGNORE_CASE`
  parameter is enabled
* Registry: Fixed duplicate UserWarning logs during model packaging
* Registry: If the huggingface pipeline text-generation model doesn't contain a default chat template, a ChatML template
  is assigned to the tokenizer.

```shell
{% for message in messages %}
  {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}
  {{ '<|im_start|>assistant\n' }}
{% endif %}"
```

* Registry: Fixed SQL queries during registry initialization that were forcing warehouse requirement

### Behavior Changes

* ML Job: The `list_jobs()` API has been modified. The `scope` parameter has been removed,
  optional `database` and `schema` parameters have been added, the return type has changed
  from `snowpark.DataFrame` to `pandas.DataFrame`, and the returned columns have been updated
  to `name`, `status`, `message`, `database_name`, `schema_name`, `owner`, `compute_pool`,
  `target_instances`, `created_time`, and `completed_time`.
* Registry: Set `relax_version` to false when pip_requirements are specified while logging model
* Registry: UserWarning will now be raised based on specified target_platforms (addresses spurious warnings)

### New Features

* Registry: `target_platforms` supports `TargetPlatformMode`: `WAREHOUSE_ONLY`, `SNOWPARK_CONTAINER_SERVICES_ONLY`,
  or `BOTH_WAREHOUSE_AND_SNOWPARK_CONTAINER_SERVICES`.
* Registry: Introduce `snowflake.ml.model.target_platform.TargetPlatform`, target platform constants, and
  `snowflake.ml.model.task.Task`.
* ML Job: Single-node ML Jobs are now in GA. Multi-node support is now in PuPr
  * Moved less frequently used job submission parameters to `**kwargs`
  * Platform metrics are now enabled by default
  * `list_jobs()` behavior changed, see [Behavior Changes](#behavior-changes) for more info

## 1.8.6

### Bug Fixes

* Fixed fatal errors from internal telemetry wrappers.

### New Features

* Registry: Add service container info to logs.
* ML Job (PuPr): Add new `submit_from_stage()` API for submitting a payload from an existing stage path.
* ML Job (PuPr): Add support for `snowpark.Session` objects in the argument list of
  `@remote` decorated functions. `Session` object will be injected from context in
  the job execution environment.

## 1.8.5

### Bug Fixes

* Registry: Fixed a bug when listing and deleting container services.
* Registry: Fixed explainability issue with scikit-learn pipelines, skipping explain function creation.
* Explainability: bump minimum streamlit version down to 1.30
* Modeling: Make XGBoost a required dependency (xgboost is not a required dependency in snowflake-ml-python 1.8.4).

### Behavior Changes

* ML Job (Multi-node PrPr): Rename argument `num_instances` to `target_instances` in job submission APIs and
  change type from `Optional[int]` to `int`

### New Features

* Registry: No longer checks if the snowflake-ml-python version is available in the Snowflake Conda channel when logging
  an SPCS-only model.
* ML Job (PuPr): Add `min_instances` argument to the job decorator to allow waiting for workers to be ready.
* ML Job (PuPr): Adjust polling behavior to reduce number of SQL calls.

### Deprecations

* `SnowflakeLoginOptions` is deprecated and will be removed in a future release.

## 1.8.4 (2025-05-12)

### Bug Fixes

* Registry: Default `enable_explainability` to True when the model can be deployed to Warehouse.
* Registry: Add `custom_model.partitioned_api` decorator and deprecate `partitioned_inference_api`.
* Registry: Fixed a bug when logging pytroch and tensorflow models that caused
  `UnboundLocalError: local variable 'multiple_inputs' referenced before assignment`.

### Behavior Changes

* ML Job (PuPr) Updated property `id` to be fully qualified name; Introduced new property `name`
  to represent the ML Job name
* ML Job (PuPr) Modified `list_jobs()` to return ML Job `name` instead of `id`
* Registry: Error in `log_model` if `enable_explainability` is True and model is only deployed to
   Snowpark Container Services, instead of just user warning.

### New Features

* ML Job (PuPr): Extend `@remote` function decorator, `submit_file()` and `submit_directory()` to accept `database` and
  `schema` parameters
* ML Job (PuPr): Support querying by fully qualified name in `get_job()`
* Explainability: Added visualization functions to `snowflake.ml.monitoring` to plot explanations in notebooks.
* Explainability: Support explain for categorical transforms for sklearn pipeline
* Support categorical type for `xgboost.DMatrix` inputs.

## 1.8.3

### New Features

* Registry: Default to the runtime cuda version if available when logging a GPU model in Container Runtime.
* ML Job (PuPr): Added `as_list` argument to `MLJob.get_logs()` to enable retrieving logs
  as a list of strings
* Registry: Support `ModelVersion.run_job` to run inference with a single-node Snowpark Container Services job.
* DataConnector: Removed PrPr decorators
* Registry: Default the target platform to warehouse when logging a partitioned model.

## 1.8.2

### New Features

* ML Job now available as a PuPr feature
  * Add ability to retrieve results for `@remote` decorated functions using
    new `MLJobWithResult.result()` API, which will return the unpickled result
    or raise an exception if the job execution failed.
  * Pre-created Snowpark Session is now available inside job payloads using
    `snowflake.snowpark.context.get_active_session()`
* Registry: Introducing `save_location` to `log_model` using the `options` argument.
  Users can use the `save_location` option to specify a local directory where the model files and configuration are written.
  This is useful when the default temporary directory has space limitations.

```python
reg.log_model(
    model=...,
    model_name=...,
    version_name=...,
    ...,
    options={"save_location": "./model_directory"},
)
```

* Registry: Include model dependencies in pip requirements by default when logging in Container Runtime.
* Multi-node ML Job (PrPr): Add `instance_id` argument to `get_logs` and `show_logs` method to support multi node log retrieval
* Multi-node ML Job (PrPr): Add `job.get_instance_status(instance_id=...)` API to support multi node status retrieval

## 1.8.1 (03-26-2025)

### Bug Fixes

* Registry: Fix a bug that caused `unsupported model type` error while logging a sklearn model with `score_samples`
  inference method.
* Registry: Fix a bug that model inference service creation fails on an existing and suspended service.

### New Features

* ML Job (PrPr): Update Container Runtime image version to `1.0.1`
* ML Job (PrPr): Add `enable_metrics` argument to job submission APIs to enable publishing service metrics to Event Table.
  See [Accessing Event Table service metrics](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/monitoring-services#accessing-event-table-service-metrics)
  for retrieving published metrics
  and [Costs of telemetry data collection](https://docs.snowflake.com/en/developer-guide/logging-tracing/logging-tracing-billing)
  for cost implications.
* Registry: When creating a copy of a `ModelVersion` with `log_model`, raise an exception if unsupported arguments are provided.

## 1.8.0 (03-20-2025)

### Bug Fixes

* Modeling: Fix a bug in some metrics that allowed an unsupported version of numpy to be installed
  automatically in the stored procedure, resulting in a numpy error on execution
* Registry: Fix a bug that leads to incorrect `Model is does not have _is_inference_api` error message when assigning
  a supported model as a property of a CustomModel.
* Registry: Fix a bug that inference is not working when models with more than 500 input features
  are deployed to SPCS.

### Behavior Change

* Registry: With FeatureGroupSpec support, auto inferred model signature for `transformers.Pipeline` models have been
  updated, including:
  * Signature for fill-mask task has been changed from

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="inputs", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureSpec(name="outputs", dtype=DataType.STRING),
        ],
    )
    ```

    to

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="inputs", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureGroupSpec(
                name="outputs",
                specs=[
                    FeatureSpec(name="sequence", dtype=DataType.STRING),
                    FeatureSpec(name="score", dtype=DataType.DOUBLE),
                    FeatureSpec(name="token", dtype=DataType.INT64),
                    FeatureSpec(name="token_str", dtype=DataType.STRING),
                ],
                shape=(-1,),
            ),
        ],
    )
    ```

  * Signature for token-classification task has been changed from

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="inputs", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureSpec(name="outputs", dtype=DataType.STRING),
        ],
    )
    ```

    to

    ```python
    ModelSignature(
        inputs=[FeatureSpec(name="inputs", dtype=DataType.STRING)],
        outputs=[
            FeatureGroupSpec(
                name="outputs",
                specs=[
                    FeatureSpec(name="word", dtype=DataType.STRING),
                    FeatureSpec(name="score", dtype=DataType.DOUBLE),
                    FeatureSpec(name="entity", dtype=DataType.STRING),
                    FeatureSpec(name="index", dtype=DataType.INT64),
                    FeatureSpec(name="start", dtype=DataType.INT64),
                    FeatureSpec(name="end", dtype=DataType.INT64),
                ],
                shape=(-1,),
            ),
        ],
    )
    ```

  * Signature for question-answering task when top_k is larger than 1 has been changed from

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="question", dtype=DataType.STRING),
            FeatureSpec(name="context", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureSpec(name="outputs", dtype=DataType.STRING),
        ],
    )
    ```

    to

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="question", dtype=DataType.STRING),
            FeatureSpec(name="context", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureGroupSpec(
                name="answers",
                specs=[
                    FeatureSpec(name="score", dtype=DataType.DOUBLE),
                    FeatureSpec(name="start", dtype=DataType.INT64),
                    FeatureSpec(name="end", dtype=DataType.INT64),
                    FeatureSpec(name="answer", dtype=DataType.STRING),
                ],
                shape=(-1,),
            ),
        ],
    )
    ```

  * Signature for text-classification task when top_k is `None` has been changed from

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="text", dtype=DataType.STRING),
            FeatureSpec(name="text_pair", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureSpec(name="label", dtype=DataType.STRING),
            FeatureSpec(name="score", dtype=DataType.DOUBLE),
        ],
    )
    ```

    to

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="text", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureSpec(name="label", dtype=DataType.STRING),
            FeatureSpec(name="score", dtype=DataType.DOUBLE),
        ],
    )
    ```

  * Signature for text-classification task when top_k is not `None` has been changed from

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="text", dtype=DataType.STRING),
            FeatureSpec(name="text_pair", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureSpec(name="outputs", dtype=DataType.STRING),
        ],
    )
    ```

    to

    ```python
    ModelSignature(
        inputs=[
            FeatureSpec(name="text", dtype=DataType.STRING),
        ],
        outputs=[
            FeatureGroupSpec(
                name="labels",
                specs=[
                    FeatureSpec(name="label", dtype=DataType.STRING),
                    FeatureSpec(name="score", dtype=DataType.DOUBLE),
                ],
                shape=(-1,),
            ),
        ],
    )
    ```

  * Signature for text-generation task has been changed from

    ```python
    ModelSignature(
        inputs=[FeatureSpec(name="inputs", dtype=DataType.STRING)],
        outputs=[
            FeatureSpec(name="outputs", dtype=DataType.STRING),
        ],
    )
    ```

    to

    ```python
    ModelSignature(
        inputs=[
            FeatureGroupSpec(
                name="inputs",
                specs=[
                    FeatureSpec(name="role", dtype=DataType.STRING),
                    FeatureSpec(name="content", dtype=DataType.STRING),
                ],
                shape=(-1,),
            ),
        ],
        outputs=[
            FeatureGroupSpec(
                name="outputs",
                specs=[
                    FeatureSpec(name="generated_text", dtype=DataType.STRING),
                ],
                shape=(-1,),
            )
        ],
    )
    ```

* Registry: PyTorch and TensorFlow models now expect a single tensor input/output by default when logging to Model
  Registry. To use multiple tensors (previous behavior), set `options={"multiple_inputs": True}`.

  Example with single tensor input:

  ```python
  import torch

  class TorchModel(torch.nn.Module):
      def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
          super().__init__()
          self.model = torch.nn.Sequential(
              torch.nn.Linear(n_input, n_hidden, dtype=dtype),
              torch.nn.ReLU(),
              torch.nn.Linear(n_hidden, n_out, dtype=dtype),
              torch.nn.Sigmoid(),
          )

      def forward(self, tensor: torch.Tensor) -> torch.Tensor:
          return cast(torch.Tensor, self.model(tensor))

  # Sample usage:
  data_x = torch.rand(size=(batch_size, n_input))

  # Log model with single tensor
  reg.log_model(
      model=model,
      ...,
      sample_input_data=data_x
  )

  # Run inference with single tensor
  mv.run(data_x)
  ```

  For multiple tensor inputs/outputs, use:

  ```python
  reg.log_model(
      model=model,
      ...,
      sample_input_data=[data_x_1, data_x_2],
      options={"multiple_inputs": True}
  )
  ```

* Registry: Default `enable_explainability` to False when the model can be deployed to Snowpark Container Services.

### New Features

* Registry: Added support to single `torch.Tensor`, `tensorflow.Tensor` and `tensorflow.Variable` as input or output
  data.
* Registry: Support [`xgboost.DMatrix`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix)
 datatype for XGBoost models.

## 1.7.5 (03-06-2025)

* Support Python 3.12.
* Explainability: Support native and snowflake.ml.modeling sklearn pipeline

### Bug Fixes

* Registry: Fixed a compatibility issue when using `snowflake-ml-python` 1.7.0 or greater to save a `tensorflow.keras`
  model with `keras` 2.x, if `relax_version` is set or default to True, and newer version of `snowflake-ml-python`
  is available in Snowflake Anaconda Channel, model could not be run in Snowflake. If you have such model, you could
  use the latest version of `snowflake-ml-python` and call `ModelVersion.load` to load it back, and re-log it.
  Alternatively, you can prevent this issue by setting `relax_version=False` when saving the model.
* Registry: Removed the validation that disallows data that does not have non-null values being passed to
  `ModelVersion.run`.
* ML Job (PrPr): No longer require CREATE STAGE privilege if `stage_name` points to an existing stage
* ML Job (PrPr): Fixed a bug causing some payload source and entrypoint path
  combinations to be erroneously rejected with
  `ValueError(f"{self.entrypoint} must be a subpath of {self.source}")`
* ML Job (PrPr): Fixed a bug in Ray cluster startup config which caused certain Runtime APIs to fail

### New Features

* Registry: Added support for handling Hugging Face model configurations with auto-mapping functionality.
* Registry: Added support for `keras` 3.x model with `tensorflow` and `pytorch` backend

## 1.7.4 (01-28-2025)

* FileSet: The `snowflake.ml.fileset.FileSet` has been deprecated and will be removed in a future version.
  Use [snowflake.ml.dataset.Dataset](https://docs.snowflake.com/en/developer-guide/snowflake-ml/dataset) and
  [snowflake.ml.data.DataConnector](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/data/snowflake.ml.data.data_connector.DataConnector)
  instead.
* Registry: `ModelVersion.run` on a service would require redeploying the service once account opts into nested function.

### Bug Fixes

* Registry: Fixed an issue that the hugging face pipeline is loaded using incorrect dtype.
* Registry: Fixed an issue that only 1 row is used when infer the model signature in the modeling model.

### New Features

* Add new `snowflake.ml.jobs` preview API for running headless workloads on SPCS using
  [Container Runtime for ML](https://docs.snowflake.com/en/developer-guide/snowflake-ml/container-runtime-ml)
* Added `guardrails` option to Cortex `complete` function, enabling
  [Cortex Guard](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#cortex-guard) support
* Model Monitoring: Expose Model Monitoring Python API by default.

## 1.7.3 (2025-01-08)

* Added lowercase versions of Cortex functions, added deprecation warning to Capitalized versions.
* Bumped the requirements of `fsspec` and `s3fs` to `>=2024.6.1,<2026`
* Bumped the requirement of `mlflow` to `>=2.16.0, <3`
* Registry: Support 500+ features for model registry
* Feature Store: Add support for `cluster_by` for feature views.

### Bug Fixes

* Registry: Fixed a bug when providing non-range index pandas DataFrame as the input to a `ModelVersion.run`.
* Registry: Improved random model version name generation to prevent collisions.
* Registry: Fix an issue when inferring signature or running inference with Snowpark data that has a column whose type
  is `ARRAY` and contains `NULL` value.
* Registry: `ModelVersion.run` now accepts fully qualified service name.
* Monitoring: Fix issue in SDK with creating monitors using fully qualified names.
* Registry: Fix error in log_model for any sklearn models with only data pre-processing including pre-processing only
  pipeline models due to default explainability enablement.

### New Features

* Added `user_files` argument to `Registry.log_model` for including images or any extra file with the model.
* Registry: Added support for handling Hugging Face model configurations with auto-mapping functionality
* DataConnector: Add new `DataConnector.from_sql()` constructor
* Registry: Provided new arguments to `snowflake.ml.model.model_signature.infer_signature` method to specify rows limit
  to be used when inferring the signature.

## 1.7.2 (2024-11-21)

### Bug Fixes

* Model Explainability: Fix issue that explain is enabled for scikit-learn pipeline
whose task is UNKNOWN and fails later when invoked.

### New Features

* Registry: Support asynchronous model inference service creation with the `block` option
  in `ModelVersion.create_service()` set to True by default.
* Registry: Allow specify `batch_size` when inferencing using sentence-transformers model.

## 1.7.1 (2024-11-05)

### Bug Fixes

* Registry: Null value is now allowed in the dataframe used in model signature inference. Null values will be ignored
 and others will be used to infer the signature.
* Registry: Pandas Extension DTypes (`pandas.StringDType()`, `pandas.BooleanDType()`, etc.) are now supported in model
signature inference.
* Registry: Null value is now allowed in the dataframe used to predict.
* Data: Fix missing `snowflake.ml.data.*` module exports in wheel
* Dataset: Fix missing `snowflake.ml.dataset.*` module exports in wheel.
* Registry: Fix the issue that `tf_keras.Model` is not recognized as keras model when logging.

### New Features

* Registry: Option to `enable_monitoring` set to False by default.  This will gate access to preview features of Model Monitoring.
* Model Monitoring: `show_model_monitors` Registry method.  This feature is still in Private Preview.
* Registry: Support `pd.Series` in input and output data.
* Model Monitoring: `add_monitor` Registry method.  This feature is still in Private Preview.
* Model Monitoring: `resume` and `suspend` ModelMonitor.  This feature is still in Private Preview.
* Model Monitoring: `get_monitor` Registry method.  This feature is still in Private Preview.
* Model Monitoring: `delete_monitor` Registry method.  This feature is still in Private Preview.

## 1.7.0 (10-22-2024)

### Behavior Change

* Generic: Require python >= 3.9.
* Data Connector: Update `to_torch_dataset` and `to_torch_datapipe` to add a dimension for scalar data.
This allows for more seamless integration with PyTorch `DataLoader`, which creates batches by stacking inputs of each batch.

Examples:

```python
ds = connector.to_torch_dataset(shuffle=False, batch_size=3)
```

* Input: "col1": [10, 11, 12]
  * Previous batch: array([10., 11., 12.]) with shape (3,)
  * New batch: array([[10.], [11.], [12.]]) with shape (3, 1)

* Input: "col2": [[0, 100], [1, 110], [2, 200]]
  * Previous batch: array([[  0, 100], [  1, 110], [  2, 200]]) with shape (3,2)
  * New batch: No change

* Model Registry: External access integrations are optional when creating a model inference service in
  Snowflake >= 8.40.0.
* Model Registry: Deprecate `build_external_access_integration` with `build_external_access_integrations` in
  `ModelVersion.create_service()`.

### Bug Fixes

* Registry: Updated `log_model` API to accept both signature and sample_input_data parameters.
* Feature Store: ExampleHelper uses fully qualified path for table name. change weather features aggregation from 1d to 1h.
* Data Connector: Return numpy array with appropriate object type instead of list for multi-dimensional
data from `to_torch_dataset` and `to_torch_datapipe`
* Model explainability: Incompatibility between SHAP 0.42.1 and XGB 2.1.1 resolved by using latest SHAP 0.46.0.

### New Features

* Registry: Provide pass keyworded variable length of arguments to class ModelContext. Example usage:

```python
mc = custom_model.ModelContext(
    config = 'local_model_dir/config.json',
    m1 = model1
)

class ExamplePipelineModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
      super().__init__(context)
      v = open(self.context['config']).read()
      self.bias = json.loads(v)['bias']

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
      model_output = self.context['m1'].predict(input)
      return pd.DataFrame({'output': model_output + self.bias})
```

* Model Development: Upgrade scikit-learn in UDTF backend for log_loss metric. As a result, `eps` argument is now ignored.
* Data Connector: Add the option of passing a `None` sized batch to `to_torch_dataset` for better
interoperability with PyTorch DataLoader.
* Model Registry: Support [pandas.CategoricalDtype](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html#pandas-categoricaldtype)
  * Limitations:
    * The native categorical data handling handling by XGBoost using `enable_categorical=True` is not supported.
    Instead please use [`sklearn.pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    to preprocess the categorical datatype and log the pipeline with the XGBoost model.
* Registry: It is now possible to pass `signatures` and `sample_input_data` at the same time to capture background
data from explainablity and data lineage.

## 1.6.4 (2024-10-17)

### Bug Fixes

* Registry: Fix an issue that leads to incident when using `ModelVersion.run` with service.

## 1.6.3 (2024-10-07)

* Model Registry (PrPr) has been removed.

### Bug Fixes

* Registry: Fix a bug that when package whose name does not follow PEP-508 is provided when logging the model,
  an unexpected normalization is happening.
* Registry: Fix `not a valid remote uri` error when logging mlflow models.
* Registry: Fix a bug that `ModelVersion.run` is called in a nested way.
* Registry: Fix an issue that leads to `log_model` failure when local package version contains parts other than
  base version.
* Fix issue where `sample_weights` were not being applied to search estimators.
* Model explainability: Fix bug which creates explain as a function instead of table function when enabling by default.
* Model explainability: Update lightgbm binary classification to return non-json values, from customer feedback.

### New Features

* Data: Improve `DataConnector.to_pandas()` performance when loading from Snowpark DataFrames.
* Model Registry: Allow users to set a model task while using `log_model`.
* Feature Store: FeatureView supports ON_CREATE or ON_SCHEDULE initialize mode.

## 1.6.2 (2024-09-04)

### Bug Fixes

* Modeling: Support XGBoost version that is larger than 2.

* Data: Fix multiple epoch iteration over `DataConnector.to_torch_datapipe()` DataPipes.
* Generic: Fix a bug that when an invalid name is provided to argument where fully qualified name is expected, it will
  be parsed wrongly. Now it raises an exception correctly.
* Model Explainability: Handle explanations for multiclass XGBoost classification models
* Model Explainability: Workarounds and better error handling for XGB>2.1.0 not working with SHAP==0.42.1

### New Features

* Data: Add top-level exports for `DataConnector` and `DataSource` to `snowflake.ml.data`.
* Data: Add native batching support via `batch_size` and `drop_last_batch` arguments to `DataConnector.to_torch_dataset()`
* Feature Store: update_feature_view() supports taking feature view object as argument.

## 1.6.1 (2024-08-12)

### Bug Fixes

* Feature Store: Support large metadata blob when generating dataset
* Feature Store: Added a hidden knob in FeatureView as kargs for setting customized
  refresh_mode
* Registry: Fix an error message in Model Version `run` when `function_name` is not mentioned and model has multiple
  target methods.
* Cortex inference: snowflake.cortex.Complete now only uses the REST API for streaming and the use_rest_api_experimental
  is no longer needed.
* Feature Store: Add a new API: FeatureView.list_columns() which list all column information.
* Data: Fix `DataFrame` ingestion with `ArrowIngestor`.

### New Features

* Enable `set_params` to set the parameters of the underlying sklearn estimator, if the snowflake-ml model has been fit.
* Data: Add `snowflake.ml.data.ingestor_utils` module with utility functions helpful for `DataIngestor` implementations.
* Data: Add new `to_torch_dataset()` connector to `DataConnector` to replace deprecated DataPipe.
* Registry: Option to `enable_explainability` set to True by default for XGBoost, LightGBM and CatBoost as PuPr feature.
* Registry: Option to `enable_explainability` when registering SHAP supported sklearn models.

## 1.6.0 (2024-07-29)

### Bug Fixes

* Modeling: `SimpleImputer` can impute integer columns with integer values.
* Registry: Fix an issue when providing a pandas Dataframe whose index is not starting from 0 as the input to
  the `ModelVersion.run`.

### New Features

* Feature Store: Add overloads to APIs accept both object and name/version. Impacted APIs include read_feature_view(),
  refresh_feature_view(), get_refresh_history(), resume_feature_view(), suspend_feature_view(), delete_feature_view().
* Feature Store: Add docstring inline examples for all public APIs.
* Feature Store: Add new utility class `ExampleHelper` to help with load source data to simplify public notebooks.
* Registry: Option to `enable_explainability` when registering XGBoost models as a pre-PuPr feature.
* Feature Store: add new API `update_entity()`.
* Registry: Option to `enable_explainability` when registering Catboost models as a pre-PuPr feature.
* Feature Store: Add new argument warehouse to FeatureView constructor to overwrite the default warehouse. Also add
  a new column 'warehouse' to the output of list_feature_views().
* Registry: Add support for logging model from a model version.
* Modeling: Distributed Hyperparameter Optimization now announce GA refresh version. The latest memory efficient version
  will not have the 10GB training limitation for dataset any more. To turn off, please run
  `
  from snowflake.ml.modeling._internal.snowpark_implementations import (
      distributed_hpo_trainer,
  )
  distributed_hpo_trainer.ENABLE_EFFICIENT_MEMORY_USAGE = False
  `
* Registry: Option to `enable_explainability` when registering LightGBM models as a pre-PuPr feature.
* Data: Add new `snowflake.ml.data` preview module which contains data reading utilities like `DataConnector`
  * `DataConnector` provides efficient connectors from Snowpark `DataFrame`
  and Snowpark ML `Dataset` to external frameworks like PyTorch, TensorFlow, and Pandas. Create `DataConnector`
  instances using the classmethod constructors `DataConnector.from_dataset()` and `DataConnector.from_dataframe()`.
* Data: Add new `DataConnector.from_sources()` classmethod constructor for constructing from `DataSource` objects.
* Data: Add new `ingestor_class` arg to `DataConnector` classmethod constructors for easier `DataIngestor` injection.
* Dataset: `DatasetReader` now subclasses new `DataConnector` class.
  * Add optional `limit` arg to `DatasetReader.to_pandas()`

### Behavior Changes

* Feature Store: change some positional parameters to keyword arguments in following APIs:
  * Entity(): desc.
  * FeatureView(): timestamp_col, refresh_freq, desc.
  * FeatureStore(): creation_mode.
  * update_entity(): desc.
  * register_feature_view(): block, overwrite.
  * list_feature_views(): entity_name, feature_view_name.
  * get_refresh_history(): verbose.
  * retrieve_feature_values(): spine_timestamp_col, exclude_columns, include_feature_view_timestamp_col.
  * generate_training_set(): save_as, spine_timestamp_col, spine_label_cols, exclude_columns,
    include_feature_view_timestamp_col.
  * generate_dataset(): version, spine_timestamp_col, spine_label_cols, exclude_columns,
    include_feature_view_timestamp_col, desc, output_type.

## 1.5.4 (2024-07-11)

### Bug Fixes

* Model Registry (PrPr): Fix 401 Unauthorized issue when deploying model to SPCS.
* Feature Store: Downgrades exceptions to warnings for few property setters in feature view. Now you can set
  desc, refresh_freq and warehouse for draft feature views.
* Modeling: Fix an issue with calling `OrdinalEncoder` with `categories` as a dictionary and a pandas DataFrame
* Modeling: Fix an issue with calling `OneHotEncoder` with `categories` as a dictionary and a pandas DataFrame

### New Features

* Registry: Allow overriding `device_map` and `device` when loading huggingface pipeline models.
* Registry: Add `set_alias` method to `ModelVersion` instance to set an alias to model version.
* Registry: Add `unset_alias` method to `ModelVersion` instance to unset an alias to model version.
* Registry: Add `partitioned_inference_api` allowing users to create partitioned inference functions in registered
  models. Enable model inference methods with table functions with vectorized process methods in registered models.
* Feature Store: add 3 more columns: refresh_freq, refresh_mode and scheduling_state to the result of
  `list_feature_views()`.
* Feature Store: `update_feature_view()` supports updating description.
* Feature Store: add new API `refresh_feature_view()`.
* Feature Store: add new API `get_refresh_history()`.
* Feature Store: Add `generate_training_set()` API for generating table-backed feature snapshots.
* Feature Store: Add `DeprecationWarning` for `generate_dataset(..., output_type="table")`.
* Feature Store: `update_feature_view()` supports updating description.
* Feature Store: add new API `refresh_feature_view()`.
* Feature Store: add new API `get_refresh_history()`.
* Model Development: OrdinalEncoder supports a list of array-likes for `categories` argument.
* Model Development: OneHotEncoder supports a list of array-likes for `categories` argument.

## 1.5.3 (06-17-2024)

### Bug Fixes

* Modeling: Fix an issue causing lineage information to be missing for
  `Pipeline`, `GridSearchCV` , `SimpleImputer`, and `RandomizedSearchCV`
* Registry: Fix an issue that leads to incorrect result when using pandas Dataframe with over 100, 000 rows as the input
  of `ModelVersion.run` method in Stored Procedure.

### New Features

* Registry: Add support for TIMESTAMP_NTZ model signature data type, allowing timestamp input and output.
* Dataset: Add `DatasetVersion.label_cols` and `DatasetVersion.exclude_cols` properties.

## 1.5.2 (06-10-2024)

### Bug Fixes

* Registry: Fix an issue that leads to unable to log model in store procedure.
* Modeling: Quick fix `import snowflake.ml.modeling.parameters.enable_anonymous_sproc` cannot be imported due to package
  dependency error.

## 1.5.1 (05-22-2024)

### Bug Fixes

* Dataset: Fix `snowflake.connector.errors.DataError: Query Result did not match expected number of rows` when accessing
  DatasetVersion properties when case insensitive `SHOW VERSIONS IN DATASET` check matches multiple version names.
* Dataset: Fix bug in SnowFS bulk file read when used with DuckDB
* Registry: Fixed a bug when loading old models.
* Lineage: Fix Dataset source lineage propagation through `snowpark.DataFrame` transformations

### Behavior Changes

* Feature Store: convert clear() into a private function. Also make it deletes feature views and entities only.
* Feature Store: Use NULL as default value for timestamp tag value.

### New Features

* Feature Store: Added new `snowflake.ml.feature_store.setup_feature_store()` API to assist Feature Store RBAC setup.
* Feature Store: Add `output_type` argument to `FeatureStore.generate_dataset()` to allow generating data snapshots
  as Datasets or Tables.
* Registry: `log_model`, `get_model`, `delete_model` now supports fully qualified name.
* Modeling: Supports anonymous stored procedure during fit calls so that modeling would not require sufficient
  permissions to operate on schema. Please call
  `import snowflake.ml.modeling.parameters.enable_anonymous_sproc  # noqa: F401`

## 1.5.0 (05-01-2024)

### Bug Fixes

* Registry: Fix invalid parameter 'SHOW_MODEL_DETAILS_IN_SHOW_VERSIONS_IN_MODEL' error.

### Behavior Changes

* Model Development: The behavior of `fit_transform` for all estimators is changed.
  Firstly, it will cover all the estimator that contains this function,
  secondly, the output would be the union of pandas DataFrame and snowpark DataFrame.

#### Model Registry (PrPr)

`snowflake.ml.registry.artifact` and related `snowflake.ml.model_registry.ModelRegistry` APIs have been removed.

* Removed `snowflake.ml.registry.artifact` module.
* Removed `ModelRegistry.log_artifact()`, `ModelRegistry.list_artifacts()`, `ModelRegistry.get_artifact()`
* Removed `artifacts` argument from `ModelRegistry.log_model()`

#### Dataset (PrPr)

`snowflake.ml.dataset.Dataset` has been redesigned to be backed by Snowflake Dataset entities.

* New `Dataset`s can be created with `Dataset.create()` and existing `Dataset`s may be loaded
  with `Dataset.load()`.
* `Dataset`s now maintain an immutable `selected_version` state. The `Dataset.create_version()` and
  `Dataset.load_version()` APIs return new `Dataset` objects with the requested `selected_version` state.
* Added `dataset.create_from_dataframe()` and `dataset.load_dataset()` convenience APIs as a shortcut
  to creating and loading `Dataset`s with a pre-selected version.
* `Dataset.materialized_table` and `Dataset.snapshot_table` no longer exist with `Dataset.fully_qualified_name`
  as the closest equivalent.
* `Dataset.df` no longer exists. Instead, use `DatasetReader.read.to_snowpark_dataframe()`.
* `Dataset.owner` has been moved to `Dataset.selected_version.owner`
* `Dataset.desc` has been moved to `DatasetVersion.selected_version.comment`
* `Dataset.timestamp_col`, `Dataset.label_cols`, `Dataset.feature_store_metadata`, and
  `Dataset.schema_version` have been removed.

#### Feature Store (PrPr)

* `FeatureStore.generate_dataset` argument list has been changed to match the new
`snowflake.ml.dataset.Dataset` definition

  * `materialized_table` has been removed and replaced with `name` and `version`.
  * `name` moved to first positional argument
  * `save_mode` has been removed as `merge` behavior is no longer supported. The new behavior is always `errorifexists`.

* Change feature view version type from str to `FeatureViewVersion`. It is a restricted string literal.

* Remove as_dataframe arg from FeatureStore.list_feature_views(), now always returns result as DataFrame.

* Combines few metadata tags into a new tag: SNOWML_FEATURE_VIEW_METADATA. This will make previously created feature views
not readable by new SDK.

### New Features

* Registry: Add `export` method to `ModelVersion` instance to export model files.
* Registry: Add `load` method to `ModelVersion` instance to load the underlying object from the model.
* Registry: Add `Model.rename` method to `Model` instance to rename or move a model.

#### Dataset (PrPr)

* Added Snowpark DataFrame integration using `Dataset.read.to_snowpark_dataframe()`
* Added Pandas DataFrame integration using `Dataset.read.to_pandas()`
* Added PyTorch and TensorFlow integrations using `Dataset.read.to_torch_datapipe()`
    and `Dataset.read.to_tf_dataset()` respectively.
* Added `fsspec` style file integration using `Dataset.read.files()` and `Dataset.read.filesystem()`

#### Feature Store

* use new tag_reference_internal to speed up metadata lookup.

## 1.4.1 (2024-04-18)

### New Features

* Registry: Add support for `catboost` model (`catboost.CatBoostClassifier`, `catboost.CatBoostRegressor`).
* Registry: Add support for `lightgbm` model (`lightgbm.Booster`, `lightgbm.LightGBMClassifier`, `lightgbm.LightGBMRegressor`).

### Bug Fixes

* Registry: Fix a bug that leads to relax_version option is not working.

### Behavior changes

* Feature Store: update_feature_view takes refresh_freq and warehouse as argument.

## 1.4.0 (2024-04-08)

### Bug Fixes

* Registry: Fix a bug when multiple models are being called from the same query, models other than the first one will
  have incorrect result. This fix only works for newly logged model.
* Modeling: When registering a model, only method(s) that is mentioned in `save_model` would be added to model signature
  in SnowML models.
* Modeling: Fix a bug that when n_jobs is not 1, model cannot execute methods such as
  predict, predict_log_proba, and other batch inference methods. The n_jobs would automatically
  set to 1 because vectorized udf currently doesn't support joblib parallel backend.
* Modeling: Fix a bug that batch inference methods cannot infer the datatype when the first row of data contains NULL.
* Modeling: Matches Distributed HPO output column names with the snowflake identifier.
* Modeling: Relax package versions for all Distributed HPO methods if the installed version
  is not available in the Snowflake conda channel
* Modeling: Add sklearn as required dependency for LightGBM package.

### Behavior Changes

* Registry: `apply` method is no longer by default logged when logging a xgboost model. If that is required, it could
  be specified manually when logging the model by `log_model(..., options={"target_methods": ["apply", ...]})`.
* Feature Store: register_entity returns an entity object.
* Feature Store: register_feature_view `block=true` becomes default.

### New Features

* Registry: Add support for `sentence-transformers` model (`sentence_transformers.SentenceTransformer`).
* Registry: Now version name is no longer required when logging a model. If not provided, a random human readable ID
  will be generated.

## 1.3.1 (2024-03-21)

### New Features

* FileSet: `snowflake.ml.fileset.sfcfs.SFFileSystem` can now be used in UDFs and stored procedures.

## 1.3.0 (2024-03-12)

### Bug Fixes

* Registry: Fix a bug that leads to module in `code_paths` when `log_model` cannot be correctly imported.
* Registry: Fix incorrect error message when validating input Snowpark DataFrame with array feature.
* Model Registry: Fix an issue when deploying a model to SPCS that some files do not have proper permission.
* Model Development: Relax package versions for all inference methods if the installed version
  is not available in the Snowflake conda channel

### Behavior Changes

* Registry: When running the method of a model, the value range based input validation to avoid input from overflowing
  is now optional rather than enforced, this should improve the performance and should not lead to problem for most
  kinds of model. If you want to enable this check as previous, specify `strict_input_validation=True` when
  calling `run`.
* Registry: By default `relax_version=True` when logging a model instead of using the specific local dependency versions.
  This improves dependency versioning by using versions available in Snowflake. To switch back to the previous behavior
  and use specific local dependency versions, specify `relax_version=False` when calling `log_model`.
* Model Development: The behavior of `fit_predict` for all estimators is changed.
  Firstly, it will cover all the estimator that contains this function,
  secondly, the output would be the union of pandas DataFrame and snowpark DataFrame.

### New Features

* FileSet: `snowflake.ml.fileset.sfcfs.SFFileSystem` can now be serialized with `pickle`.

## 1.2.3 (2024-02-26)

### Bug Fixes

* Registry: Now when providing Decimal Type column to a DOUBLE or FLOAT feature will not error out but auto cast with
  warnings.
* Registry: Improve the error message when specifying currently unsupported `pip_requirements` argument.
* Model Development: Fix precision_recall_fscore_support incorrect results when `average="samples"`.
* Model Registry: Fix an issue that leads to description, metrics or tags are not correctly returned in newly created
  Model Registry (PrPr) due to Snowflake BCR [2024_01](https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_01/bcr-1483)

### Behavior Changes

* Feature Store: `FeatureStore.suspend_feature_view` and `FeatureStore.resume_feature_view` doesn't mutate input feature
  view argument any more. The updated status only reflected in the returned feature view object.

### New Features

* Model Development: support `score_samples` method for all the classes, including Pipeline,
  GridSearchCV, RandomizedSearchCV, PCA, IsolationForest, ...
* Registry: Support deleting a version of a model.

## 1.2.2 (2024-02-13)

### New Features

* Model Registry: Support providing external access integrations when deploying a model to SPCS. This will help and be
  required to make sure the deploying process work as long as SPCS will by default deny all network connections. The
  following endpoints must be allowed to make deployment work: docker.com:80, docker.com:443, anaconda.com:80,
  anaconda.com:443, anaconda.org:80, anaconda.org:443, pypi.org:80, pypi.org:443. If you are using
  `snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` object, the following endpoints are required
  to be allowed: huggingface.com:80, huggingface.com:443, huggingface.co:80, huggingface.co:443.

## 1.2.1 (2024-01-25)

### New Features

* Model Development: Infers output column data type for transformers when possible.
* Registry: `relax_version` option is available in the `options` argument when logging the model.

## 1.2.0 (2024-01-11)

### Bug Fixes

* Model Registry: Fix "XGBoost version not compiled with GPU support" error when running CPU inference against open-source
  XGBoost models deployed to SPCS.
* Model Registry: Fix model deployment to SPCS on Windows machines.

### New Features

* Model Development: Introduced XGBoost external memory training feature. This feature enables training XGBoost models
  on large datasets that don't fit into memory.
* Registry: New Registry class named `snowflake.ml.registry.Registry` providing similar APIs as the old one but works
  with new MODEL object in Snowflake SQL. Also, we are providing`snowflake.ml.model.Model` and
  `snowflake.ml.model.ModelVersion` to represent a model and a specific version of a model.
* Model Development: Add support for `fit_predict` method in `AgglomerativeClustering`, `DBSCAN`, and `OPTICS` classes;
* Model Development: Add support for `fit_transform` method in `MDS`, `SpectralEmbedding` and `TSNE` class.

### Additional Notes

* Model Registry: The `snowflake.ml.registry.model_registry.ModelRegistry` has been deprecated starting from version
  1.2.0. It will stay in the Private Preview phase. For future implementations, kindly utilize
  `snowflake.ml.registry.Registry`, except when specifically required. The old model registry will be removed once all
  its primary functionalities are fully integrated into the new registry.

## 1.1.2 (2023-12-18)

### Bug Fixes

* Generic: Fix the issue that stack trace is hidden by telemetry unexpectedly.
* Model Development: Execute model signature inference without materializing full dataframe in memory.
* Model Registry: Fix occasional 'snowflake-ml-python library does not exist' error when deploying to SPCS.

### Behavior Changes

* Model Registry: When calling `predict` with Snowpark DataFrame, both inferred or normalized column names are accepted.
* Model Registry: When logging a Snowpark ML Modeling Model, sample input data or manually provided signature will be
  ignored since they are not necessary.

### New Features

* Model Development: SQL implementation of binary `precision_score` metric.

## 1.1.1 (2023-12-05)

### Bug Fixes

* Model Registry: The `predict` target method on registered models is now compatible with unsupervised estimators.
* Model Development: Fix confusion_matrix incorrect results when the row number cannot be divided by the batch size.

### New Features

* Introduced passthrough_col param in Modeling API. This new param is helpful in scenarios
  requiring automatic input_cols inference, but need to avoid using specific
  columns, like index columns, during training or inference.

## 1.1.0 (2023-12-01)

### Bug Fixes

* Model Registry: Fix panda dataframe input not handling first row properly.
* Model Development: OrdinalEncoder and LabelEncoder output_columns do not need to be valid snowflake identifiers. They
  would previously be excluded if the normalized name did not match the name specified in output_columns.

### New Features

* Model Registry: Add support for invoking public endpoint on SPCS service, by providing a "enable_ingress" SPCS
  deployment option.
* Model Development: Add support for distributed HPO - GridSearchCV and RandomizedSearchCV execution will be
  distributed on multi-node warehouses.

## 1.0.12 (2023-11-13)

### Bug Fixes

* Model Registry: Fix regression issue that container logging is not shown during model deployment to SPCS.
* Model Development: Enhance the column capacity of OrdinalEncoder.
* Model Registry: Fix unbound `batch_size` error when deploying a model other than Hugging Face Pipeline
  and LLM with GPU on SPCS.

### Behavior Changes

* Model Registry: Raise early error when deploying to SPCS with db/schema that starts with underscore.
* Model Registry: `conda-forge` channel is now automatically added to channel lists when deploying to SPCS.
* Model Registry: `relax_version` will not strip all version specifier, instead it will relax `==x.y.z` specifier to
  `>=x.y,<(x+1)`.
* Model Registry: Python with different patchlevel but the same major and minor will not result a warning when loading
  the model via Model Registry and would be considered to use when deploying to SPCS.
* Model Registry: When logging a `snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` object,
  versions of local installed libraries won't be picked as dependencies of models, instead it will pick up some pre-
  defined dependencies to improve user experience.

### New Features

* Model Registry: Enable best-effort SPCS job/service log streaming when logging level is set to INFO.

## 1.0.11 (2023-10-27)

### New Features

* Model Registry: Add log_artifact() public method.
* Model Development: Add support for `kneighbors`.

### Behavior Changes

* Model Registry: Change log_model() argument from TrainingDataset to List of Artifact.
* Model Registry: Change get_training_dataset() to get_artifact().

### Bug Fixes

* Model Development: Fix support for XGBoost and LightGBM models using SKLearn Grid Search and Randomized Search model selectors.
* Model Development: DecimalType is now supported as a DataType.
* Model Development: Fix metrics compatibility with Snowpark Dataframes that use Snowflake identifiers
* Model Registry: Resolve 'delete_deployment' not deleting the SPCS service in certain cases.

## 1.0.10 (2023-10-13)

### Behavior Changes

* Model Development: precision_score, recall_score, f1_score, fbeta_score, precision_recall_fscore_support,
  mean_absolute_error, mean_squared_error, and mean_absolute_percentage_error metric calculations are now distributed.
* Model Registry: `deploy` will now return `Deployment` for deployment information.

### New Features

* Model Registry: When the model signature is auto-inferred, it will be printed to the log for reference.
* Model Registry: For SPCS deployment, `Deployment` details will contains `image_name`, `service_spec` and `service_function_sql`.

### Bug Fixes

* Model Development: Fix an issue that leading to UTF-8 decoding errors when using modeling modules on Windows.
* Model Development: Fix an issue that alias definitions cause `SnowparkSQLUnexpectedAliasException` in inference.
* Model Registry: Fix an issue that signature inference could be incorrect when using Snowpark DataFrame as sample input.
* Model Registry: Fix too strict data type validation when predicting. Now, for example, if you have a INT8
  type feature in the signature, if providing a INT64 dataframe but all values are within the range, it would not fail.

## 1.0.9 (2023-09-28)

### Behavior Changes

* Model Development: log_loss metric calculation is now distributed.

### Bug Fixes

* Model Registry: Fix an issue that building images fails with specific docker setup.
* Model Registry: Fix an issue that unable to embed local ML library when the library is imported by `zipimport`.
* Model Registry: Fix out-of-date doc about `platform` argument in the `deploy` function.
* Model Registry: Fix an issue that unable to deploy a GPU-trained PyTorch model to a platform where GPU is not available.

## 1.0.8 (2023-09-15)

### Bug Fixes

* Model Development: Ordinal encoder can be used with mixed input column types.
* Model Development: Fix an issue when the sklearn default value is `np.nan`.
* Model Registry: Fix an issue that incorrect docker executable is used when building images.
* Model Registry: Fix an issue that specifying `token` argument when using
  `snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` with `transformers < 4.32.0` is not effective.
* Model Registry: Fix an issue that incorrect system function call is used when deploying to SPCS.
* Model Registry: Fix an issue when using a `transformers.pipeline` that does not have a `tokenizer`.
* Model Registry: Fix incorrectly-inferred image repository name during model deployment to SPCS.
* Model Registry: Fix GPU resource retention issue caused by failed or stuck previous deployments in SPCS.

## 1.0.7 (2023-09-05)

### Bug Fixes

* Model Development & Model Registry: Fix an error related to `pandas.io.json.json_normalize`.
* Allow disabling telemetry.

## 1.0.6 (2023-09-01)

### New Features

* Model Registry: add `create_if_not_exists` parameter in constructor.
* Model Registry: Added get_or_create_model_registry API.
* Model Registry: Added support for using GPU inference when deploying XGBoost (`xgboost.XGBModel` and `xgboost.Booster`
  ), PyTorch (`torch.nn.Module` and `torch.jit.ScriptModule`) and TensorFlow (`tensorflow.Module` and
  `tensorflow.keras.Model`) models to Snowpark Container Services.
* Model Registry: When inferring model signature, `Sequence` of built-in types, `Sequence` of `numpy.ndarray`,
  `Sequence` of `torch.Tensor`, `Sequence` of `tensorflow.Tensor` and `Sequence` of `tensorflow.Tensor` can be used
  instead of only `List` of them.
* Model Registry: Added `get_training_dataset` API.
* Model Development: Size of metrics result can exceed previous 8MB limit.
* Model Registry: Added support save/load/deploy HuggingFace pipeline object (`transformers.Pipeline`) and our wrapper
  (`snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel`) to it. Using the wrapper to specify
  configurations and the model for the pipeline will be loaded dynamically when deploying. Currently, following tasks
  are supported to log without manually specifying model signatures:
  * "conversational"
  * "fill-mask"
  * "question-answering"
  * "summarization"
  * "table-question-answering"
  * "text2text-generation"
  * "text-classification" (alias "sentiment-analysis" available)
  * "text-generation"
  * "token-classification" (alias "ner" available)
  * "translation"
  * "translation_xx_to_yy"
  * "zero-shot-classification"

### Bug Fixes

* Model Development: Fixed a bug when using simple imputer with numpy >= 1.25.
* Model Development: Fixed a bug when inferring the type of label columns.

### Behavior Changes

* Model Registry: `log_model()` now return a `ModelReference` object instead of a model ID.
* Model Registry: When deploying a model with 1 `target method` only, the `target_method` argument can be omitted.
* Model Registry: When using the snowflake-ml-python with version newer than what is available in Snowflake Anaconda
  Channel, `embed_local_ml_library` option will be set as `True` automatically if not.
* Model Registry: When deploying a model to Snowpark Container Services and using GPU, the default value of num_workers
  will be 1.
* Model Registry: `keep_order` and `output_with_input_features` in the deploy options have been removed. Now the
  behavior is controlled by the type of the input when calling `model.predict()`. If the input is a `pandas.DataFrame`,
  the behavior will be the same as `keep_order=True` and `output_with_input_features=False` before. If the input is a
  `snowpark.DataFrame`, the behavior will be the same as `keep_order=False` and `output_with_input_features=True` before.
* Model Registry: When logging and deploying PyTorch (`torch.nn.Module` and `torch.jit.ScriptModule`) and TensorFlow
  (`tensorflow.Module` and `tensorflow.keras.Model`) models, we no longer accept models whose input is a list of tensor
  and output is a list of tensors. Instead, now we accept models whose input is 1 or more tensors as positional arguments,
  and output is a tensor or a tuple of tensors. The input and output dataframe when predicting keep the same as before,
  that is every column is an array feature and contains a tensor.

## 1.0.5 (2023-08-17)

### New Features

* Model Registry: Added support save/load/deploy xgboost Booster model.
* Model Registry: Added support to get the model name and the model version from model references.

### Bug Fixes

* Model Registry: Restore the db/schema back to the session after `create_model_registry()`.
* Model Registry: Fixed an issue that the UDF name created when deploying a model is not identical to what is provided
  and cannot be correctly dropped when deployment getting dropped.
* connection_params.SnowflakeLoginOptions(): Added support for `private_key_path`.

## 1.0.4 (2023-07-28)

### New Features

* Model Registry: Added support save/load/deploy Tensorflow models (`tensorflow.Module`).
* Model Registry: Added support save/load/deploy MLFlow PyFunc models (`mlflow.pyfunc.PyFuncModel`).
* Model Development: Input dataframes can now be joined against data loaded from staged files.
* Model Development: Added support for non-English languages.

### Bug Fixes

* Model Registry: Fix an issue that model dependencies are incorrectly reported as unresolvable on certain platforms.

## 1.0.3 (2023-07-14)

### Behavior Changes

* Model Registry: When predicting a model whose output is a list of NumPy ndarray, the output would not be flattened,
  instead, every ndarray will act as a feature(column) in the output.

### New Features

* Model Registry: Added support save/load/deploy PyTorch models (`torch.nn.Module` and `torch.jit.ScriptModule`).

### Bug Fixes

* Model Registry: Fix an issue that when database or schema name provided to `create_model_registry` contains special
  characters, the model registry cannot be created.
* Model Registry: Fix an issue that `get_model_description` returns with additional quotes.
* Model Registry: Fix incorrect error message when attempting to remove a unset tag of a model.
* Model Registry: Fix a typo in the default deployment table name.
* Model Registry: Snowpark dataframe for sample input or input for `predict` method that contains a column with
  Snowflake `NUMBER(precision, scale)` data type where `scale = 0` will not lead to error, and will now correctly
  recognized as `INT64` data type in model signature.
* Model Registry: Fix an issue that prevent model logged in the system whose default encoding is not UTF-8 compatible
  from deploying.
* Model Registry: Added earlier and better error message when any file name in the model or the file name of model
  itself contains characters that are unable to be encoded using ASCII. It is currently not supported to deploy such a
  model.

## 1.0.2 (2023-06-22)

### Behavior Changes

* Model Registry: Prohibit non-snowflake-native models from being logged.
* Model Registry: `_use_local_snowml` parameter in options of `deploy()` has been removed.
* Model Registry: A default `False` `embed_local_ml_library` parameter has been added to the options of `log_model()`.
  With this set to `False` (default), the version of the local snowflake-ml-python library will be recorded and used when
  deploying the model. With this set to `True`, local snowflake-ml-python library will be embedded into the logged model,
  and will be used when you load or deploy the model.

### New Features

* Model Registry: A new optional argument named `code_paths` has been added to the arguments of `log_model()` for users
  to specify additional code paths to be imported when loading and deploying the model.
* Model Registry: A new optional argument named `options` has been added to the arguments of `log_model()` to specify
  any additional options when saving the model.
* Model Development: Added metrics:
  * d2_absolute_error_score
  * d2_pinball_score
  * explained_variance_score
  * mean_absolute_error
  * mean_absolute_percentage_error
  * mean_squared_error

### Bug Fixes

* Model Development: `accuracy_score()` now works when given label column names are lists of a single value.

## 1.0.1 (2023-06-16)

### Behavior Changes

* Model Development: Changed Metrics APIs to imitate sklearn metrics modules:
  * `accuracy_score()`, `confusion_matrix()`, `precision_recall_fscore_support()`, `precision_score()` methods move from
    respective modules to `metrics.classification`.
* Model Registry: The default table/stage created by the Registry now uses "_SYSTEM_" as a prefix.
* Model Registry: `get_model_history()` method as been enhanced to include the history of model deployment.

### New Features

* Model Registry: A default `False` flag named `replace_udf` has been added to the options of `deploy()`. Setting this
  to `True` will allow overwrite existing UDF with the same name when deploying.
* Model Development: Added metrics:
  * f1_score
  * fbeta_score
  * recall_score
  * roc_auc_score
  * roc_curve
  * log_loss
  * precision_recall_curve
* Model Registry: A new argument named `permanent` has been added to the argument of `deploy()`. Setting this to `True`
  allows the creation of a permanent deployment without needing to specify the UDF location.
* Model Registry: A new method `list_deployments()` has been added to enumerate all permanent deployments originating
  from a specific model.
* Model Registry: A new method `get_deployment()` has been added to fetch a deployment by its deployment name.
* Model Registry: A new method `delete_deployment()` has been added to remove an existing permanent deployment.

## 1.0.0 (2023-06-09)

### Behavior Changes

* Model Registry: `predict()` method moves from Registry to ModelReference.
* Model Registry: `_snowml_wheel_path` parameter in options of `deploy()`, is replaced with `_use_local_snowml` with
  default value of `False`. Setting this to `True` will have the same effect of uploading local SnowML code when executing
  model in the warehouse.
* Model Registry: Removed `id` field from `ModelReference` constructor.
* Model Development: Preprocessing and Metrics move to the modeling package: `snowflake.ml.modeling.preprocessing` and
  `snowflake.ml.modeling.metrics`.
* Model Development: `get_sklearn_object()` method is renamed to `to_sklearn()`, `to_xgboost()`, and `to_lightgbm()` for
  respective native models.

### New Features

* Added PolynomialFeatures transformer to the snowflake.ml.modeling.preprocessing module.
* Added metrics:
  * accuracy_score
  * confusion_matrix
  * precision_recall_fscore_support
  * precision_score

### Bug Fixes

* Model Registry: Model version can now be any string (not required to be a valid identifier)
* Model Deployment: `deploy()` & `predict()` methods now correctly escapes identifiers

## 0.3.2 (2023-05-23)

### Behavior Changes

* Use cloudpickle to serialize and deserialize models throughout the codebase and removed dependency on joblib.

### New Features

* Model Deployment: Added support for snowflake.ml models.

## 0.3.1 (2023-05-18)

### Behavior Changes

* Standardized registry API with following
  * Create & open registry taking same set of arguments
  * Create & Open can choose schema to use
  * Set_tag, set_metric, etc now explicitly calls out arg name as metric_name, tag_name, metric_name, etc.

### New Features

* Changes to support python 3.9, 3.10
* Added kBinsDiscretizer
* Support for deployment of XGBoost models & int8 types of data

## 0.3.0 (2023-05-11)

### Behavior Changes

* Big Model Registry Refresh
  * Fixed API discrepancies between register_model & log_model.
  * Model can be referred by Name + Version (no opaque internal id is required)

### New Features

* Model Registry: Added support save/load/deploy SKL & XGB Models

## 0.2.3 (2023-04-27)

### Bug Fixes

* Allow using OneHotEncoder along with sklearn style estimators in a pipeline.

### New Features

* Model Registry: Added support for delete_model. Use delete_artifact = False to not delete the underlying model data
  but just unregister.

## 0.2.2 (2023-04-11)

### New Features

* Initial version of snowflake-ml modeling package.
  * Provide support for training most of scikit-learn and xgboost estimators and transformers.

### Bug Fixes

* Minor fixes in preprocessing package.

## 0.2.1 (2023-03-23)

### New Features

* New in Preprocessing:
  * SimpleImputer
  * Covariance Matrix
* Optimization of Ordinal Encoder client computations.

### Bug Fixes

* Minor fixes in OneHotEncoder.

## 0.2.0 (2023-02-27)

### New Features

* Model Registry
* PyTorch & Tensorflow connector file generic FileSet API
* New to Preprocessing:
  * Binarizer
  * Normalizer
  * Pearson correlation Matrix
* Optimization in Ordinal Encoder to cache vocabulary in temp tables.

## 0.1.3 (2023-02-02)

### New Features

* Initial version of transformers including:
  * Label Encoder
  * Max Abs Scaler
  * Min Max Scaler
  * One Hot Encoder
  * Ordinal Encoder
  * Robust Scaler
  * Standard Scaler
