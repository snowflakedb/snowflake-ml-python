# Snowpark ML

Snowpark ML is a set of tools including SDKs and underlying infrastructure to build and deploy machine learning models.
With Snowpark ML, you can pre-process data, train, manage and deploy ML models all within Snowflake, using a single SDK,
and benefit from Snowflake’s proven performance, scalability, stability and governance at every stage of the Machine
Learning workflow.

## Key Components of Snowpark ML

The Snowpark ML Python SDK provides a number of APIs to support each stage of an end-to-end Machine Learning development
and deployment process, and includes two key components.

### Snowpark ML Development

[Snowpark ML Development](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#ml-modeling)
provides a collection of python APIs enabling efficient ML model development directly in Snowflake:

1. Modeling API (`snowflake.ml.modeling`) for data preprocessing, feature engineering and model training in Snowflake.
This includes the `snowflake.ml.modeling.preprocessing` module for scalable data transformations on large data sets
utilizing the compute resources of underlying Snowpark Optimized High Memory Warehouses, and a large collection of ML
model development classes based on sklearn, xgboost, and lightgbm.

1. Framework Connectors: Optimized, secure and performant data provisioning for Pytorch and Tensorflow frameworks in
their native data loader formats.

1. FileSet API: FileSet provides a Python fsspec-compliant API for materializing data into a Snowflake internal stage
from a query or Snowpark Dataframe along with a number of convenience APIs.

### Snowflake MLOps

Snowflake MLOps contains suit of tools and objects to make ML development cycle. It complements
the Snowpark ML Development API, and provides end to end development to deployment within Snowflake.
Currently, the API consists of:

1. [Registry](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#snowflake-model-registry): A python API
  allows secure deployment and management of models in Snowflake, supporting models trained both inside and outside of
  Snowflake.
2. [Feature Store](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#snowflake-feature-store): A fully
  integrated solution for defining, managing, storing and discovering ML features derived from your data. The
  Snowflake Feature Store supports automated, incremental refresh from batch and streaming data sources, so that
  feature pipelines need be defined only once to be continuously updated with new data.
3. [Datasets](https://docs.snowflake.com/developer-guide/snowflake-ml/overview#snowflake-datasets): Dataset provide an
  immutable, versioned snapshot of your data suitable for ingestion by your machine learning models.

## Getting started

### Have your Snowflake account ready

If you don't have a Snowflake account yet, you can [sign up for a 30-day free trial account](https://signup.snowflake.com/).

### Installation

Follow the [installation instructions](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#installing-snowpark-ml)
in the Snowflake documentation.

Python versions 3.9 to 3.12 are supported. You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[anaconda](https://www.anaconda.com/) to create a Conda environment (recommended),
or [virtualenv](https://docs.python.org/3/tutorial/venv.html) to create a virtual environment.

### Conda channels

The [Snowflake Conda Channel](https://repo.anaconda.com/pkgs/snowflake/) contains the official snowpark ML package releases.
The recommended approach is to install `snowflake-ml-python` this conda channel:

```sh
conda install \
  -c https://repo.anaconda.com/pkgs/snowflake \
  --override-channels \
  snowflake-ml-python
```

See [the developer guide](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index) for installation instructions.

The latest version of the `snowpark-ml-python` package is also published in a conda channel in this repository. Package versions
in this channel may not yet be present in the official Snowflake conda channel.

Install `snowflake-ml-python` from this channel with the following (being sure to replace `<version_specifier>` with the
desired version, e.g. `1.0.10`):

```bash
conda install \
  -c https://raw.githubusercontent.com/snowflakedb/snowflake-ml-python/conda/releases/  \
  -c https://repo.anaconda.com/pkgs/snowflake \
  --override-channels \
  snowflake-ml-python==<version_specifier>
```

Note that until a `snowflake-ml-python` package version is available in the official Snowflake conda channel, there may
be compatibility issues. Server-side functionality that `snowflake-ml-python` depends on may not yet be released.

### Verifying the package

1. Install cosign.
  This example is using golang installation: [installing-cosign-with-go](https://edu.chainguard.dev/open-source/sigstore/cosign/how-to-install-cosign/#installing-cosign-with-go).
1. Download the file from the repository like [pypi](https://pypi.org/project/snowflake-ml-python/#files).
1. Download the signature files from the [release tag](https://github.com/snowflakedb/snowflake-ml-python/releases/tag/1.7.0).
1. Verify signature on projects signed using Jenkins job:

   ```sh
   cosign verify-blob snowflake_ml_python-1.7.0.tar.gz --key snowflake-ml-python-1.7.0.pub --signature resources.linux.snowflake_ml_python-1.7.0.tar.gz.sig

   cosign verify-blob snowflake_ml_python-1.7.0.tar.gz --key snowflake-ml-python-1.7.0.pub --signature resources.linux.snowflake_ml_python-1.7.0
   ```

NOTE: Version 1.7.0 is used as example here. Please choose the the latest version.
