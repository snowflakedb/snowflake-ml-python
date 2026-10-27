# Snowflake ML Python

`snowflake-ml-python` is the Python library for AI and machine learning on Snowflake. Use it to build and manage
features with the Snowflake Feature Store, train with the open-source frameworks you already use, and manage, serve,
and monitor models on Snowflake compute.

Train machine learning models in the
[Snowflake Container Runtime](https://docs.snowflake.com/en/developer-guide/snowflake-ml/container-runtime-ml), a
managed environment for notebooks and batch workloads on CPU or GPU, with common ML packages already available. Work in
[Snowflake Notebooks](https://docs.snowflake.com/en/developer-guide/snowflake-ml/notebooks-on-spcs), or send the same
workloads to the Container Runtime from your own IDE.

See the [Snowflake ML developer guide](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview) for the full
platform.

## APIs

- **[Model Registry](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview)**
  (`snowflake.ml.registry`). Log and manage versions of models trained in Snowflake or elsewhere, including
  scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow, Keras, Hugging Face, and MLflow models, as well as your own
  Python models. Run inference in a warehouse, or deploy the model to Snowpark Container Services.

- **[Feature Store](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)**
  (`snowflake.ml.feature_store`). Define, discover, and maintain features derived from your data. Pipelines refresh
  from batch and streaming sources, and the same definitions serve both training and online inference.

- **[ML Jobs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/overview)** (`snowflake.ml.jobs`).
  Run training and other Python ML workloads on Container Runtime from VS Code, Jupyter, or a pipeline. Submit a file,
  a directory, or a function.

- **[Experiment tracking](https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiments)**
  (`snowflake.ml.experiment`). Log parameters, metrics, and models while you train, then compare runs and choose what
  to deploy.

- **[Datasets](https://docs.snowflake.com/en/developer-guide/snowflake-ml/dataset)** (`snowflake.ml.dataset`). Keep
  immutable, versioned snapshots of training data, and read them from Snowpark, PyTorch, or TensorFlow.

- **[ML Observability](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/model-observability)**
  (`snowflake.ml.monitoring`). Monitor performance and drift for models in the registry, and explain model
  predictions.

- **[ML Lineage](https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-lineage)** (`snowflake.ml.lineage`).
  Trace source data through features, datasets, and models.

- **[Modeling and preprocessing](https://docs.snowflake.com/en/developer-guide/snowflake-ml/transform-data)**
  (`snowflake.ml.modeling`). `snowflake.ml.modeling.preprocessing` runs transformations such as scaling and encoding
  in a Snowflake warehouse. The package also includes scikit-learn, XGBoost, and LightGBM estimators.

To load Snowflake tables and stages into pandas, PyTorch, or TensorFlow while training in Container Runtime, use the
[Data Connector](https://docs.snowflake.com/en/developer-guide/snowflake-ml/load-data) (`snowflake.ml.data`).

## Getting started

If you don't have a Snowflake account yet, you can [sign up for a 30-day free trial](https://signup.snowflake.com/).

### Installation

Python 3.10 through 3.14 are supported.

`snowflake-ml-python` is preinstalled in
[Container Runtime](https://docs.snowflake.com/en/developer-guide/snowflake-ml/container-runtime-ml). The version in the
image depends on the runtime you select. Each runtime's package list is in the [Container Runtime release
notes](https://docs.snowflake.com/en/developer-guide/snowflake-ml/container-runtime/releases).

In warehouse notebooks and worksheets, select `snowflake-ml-python` from the Packages menu. The package and its
dependencies must be allowed by your organization's package policy. See
[Python APIs for Snowflake ML](https://docs.snowflake.com/en/developer-guide/snowflake-ml/snowpark-ml).

Locally, install from PyPI:

```sh
python -m pip install snowflake-ml-python
```

Some libraries are optional extras rather than part of the base install. For example:

```sh
python -m pip install "snowflake-ml-python[scikit-learn]"
python -m pip install "snowflake-ml-python[feature_store]"  # online feature reads
python -m pip install "snowflake-ml-python[all]"
```

The [Snowflake Anaconda channel](https://repo.anaconda.com/pkgs/snowflake/) is the preferred conda source. Packages
there are validated with Snowflake ML:

```sh
conda create --name snowflake-ml
conda activate snowflake-ml
conda install \
  --override-channels \
  --channel https://repo.anaconda.com/pkgs/snowflake/ \
  snowflake-ml-python
```

The package is also on [conda-forge](https://anaconda.org/conda-forge/snowflake-ml-python):

```sh
conda install \
  --override-channels \
  --channel https://conda.anaconda.org/conda-forge/ \
  snowflake-ml-python
```

Conda does not support pip extras. Install optional libraries such as `scikit-learn`, `xgboost`, or `httpx` (online
Feature Store reads) as separate conda packages.

### Verifying a release

Release artifacts on PyPI are signed with [cosign](https://docs.sigstore.dev/cosign/system_config/installation/). For a
given version, download the sdist or wheel from [PyPI](https://pypi.org/project/snowflake-ml-python/#files) and the
`.pub` and `.sig` files from the matching [GitHub release](https://github.com/snowflakedb/snowflake-ml-python/releases).

```sh
VERSION=2.5.0  # use the version you downloaded

cosign verify-blob \
  "snowflake_ml_python-${VERSION}.tar.gz" \
  --key "snowflake-ml-python-${VERSION}.pub" \
  --signature "resources.linux.snowflake_ml_python-${VERSION}.tar.gz.sig"

cosign verify-blob \
  "snowflake_ml_python-${VERSION}-py3-none-any.whl" \
  --key "snowflake-ml-python-${VERSION}.pub" \
  --signature "resources.linux.snowflake_ml_python-${VERSION}-py3-none-any.whl.sig"
```
