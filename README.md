# Snowflake ML Python

Snowflake ML Python is a set of tools including SDKs and underlying infrastructure to build and deploy machine learning models.
With Snowflake ML Python, you can pre-process data, train, manage and deploy ML models all within Snowflake,
and benefit from Snowflake’s proven performance, scalability, stability and governance at every stage of the Machine
Learning workflow.

## Key Components of Snowflake ML Python

The Snowflake ML Python SDK provides a number of APIs to support each stage of an end-to-end Machine Learning development
and deployment process.

### Snowflake ML Model Development

[Snowflake ML Model Development](https://docs.snowflake.com/developer-guide/snowflake-ml/overview#ml-modeling)
provides a collection of python APIs enabling efficient ML model development directly in Snowflake:

1. Modeling API (`snowflake.ml.modeling`) for data preprocessing, feature engineering and model training in Snowflake.
This includes the `snowflake.ml.modeling.preprocessing` module for scalable data transformations on large data sets
utilizing the compute resources of underlying Snowpark Optimized High Memory Warehouses, and a large collection of ML
model development classes based on sklearn, xgboost, and lightgbm.

1. Framework Connectors: Optimized, secure and performant data provisioning for Pytorch and Tensorflow frameworks in
their native data loader formats.

### Snowflake ML Ops

Snowflake ML Python contains a suite of MLOps tools. It complements
the Snowflake Modeling API, and provides end to end development to deployment within Snowflake.
The Snowflake ML Ops suite consists of:

1. [Registry](https://docs.snowflake.com/developer-guide/snowflake-ml/overview#snowflake-model-registry): A python API
  allows secure deployment and management of models in Snowflake, supporting models trained both inside and outside of
  Snowflake.
2. [Feature Store](https://docs.snowflake.com/developer-guide/snowflake-ml/overview#snowflake-feature-store): A fully
  integrated solution for defining, managing, storing and discovering ML features derived from your data. The
  Snowflake Feature Store supports automated, incremental refresh from batch and streaming data sources, so that
  feature pipelines need be defined only once to be continuously updated with new data.
3. [Datasets](https://docs.snowflake.com/developer-guide/snowflake-ml/overview#snowflake-datasets): Dataset provide an
  immutable, versioned snapshot of your data suitable for ingestion by your machine learning models.

## Getting started

Learn about all Snowflake ML feature offerings in the [Developer Guide](https://docs.snowflake.com/developer-guide/snowflake-ml/overview).

### Have your Snowflake account ready

If you don't have a Snowflake account yet, you can [sign up for a 30-day free trial account](https://signup.snowflake.com/).

### Installation

Snowflake ML Python is pre-installed in Container Runtime notebook environments.
[Learn more](https://docs.snowflake.com/en/developer-guide/snowflake-ml/notebooks-on-spcs).

In Snowflake Warehouse notebook environments, snowflake-ml-python can be installed using the "Packages" drop-down menu.

Follow the [installation instructions](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#installing-snowpark-ml)
in the Snowflake documentation.

Python versions 3.9 to 3.12 are supported. You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[anaconda](https://www.anaconda.com/) to create a Conda environment (recommended),
or [virtualenv](https://docs.python.org/3/tutorial/venv.html) to create a virtual environment.

### Conda channels

The [Snowflake Anaconda Channel](https://repo.anaconda.com/pkgs/snowflake/) contains the official snowflake-ml-python package
releases. To install `snowflake-ml-python` from this conda channel:

```sh
conda install \
  -c https://repo.anaconda.com/pkgs/snowflake \
  --override-channels \
  snowflake-ml-python
```

See [the developer guide](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index) for detailed installation instructions.

The snowflake-ml-python package is also published in [conda-forge](https://anaconda.org/conda-forge/snowflake-ml-python).
To install `snowflake-ml-python` from conda forge:

```sh
conda install \
  -c https://conda.anaconda.org/conda-forge/ \
  --override-channels \
  snowflake-ml-python
```

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
