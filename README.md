# Snowpark ML

Snowpark ML is a set of tools including SDKs and underlying infrastructure to build and deploy machine learning models.
With Snowpark ML, you can pre-process data, train, manage and deploy ML models all within Snowflake, using a single SDK,
and benefit from Snowflake’s proven performance, scalability, stability and governance at every stage of the Machine
Learning workflow.

## Key Components of Snowpark ML

The Snowpark ML Python SDK provides a number of APIs to support each stage of an end-to-end Machine Learning development
and deployment process, and includes two key components.

### Snowpark ML Development

[Snowpark ML Development](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#snowpark-ml-development)
provides a collection of python APIs enabling efficient ML model development directly in Snowflake:

1. Modeling API (`snowflake.ml.modeling`) for data preprocessing, feature engineering and model training in Snowflake.
This includes the `snowflake.ml.modeling.preprocessing` module for scalable data transformations on large data sets
utilizing the compute resources of underlying Snowpark Optimized High Memory Warehouses, and a large collection of ML
model development classes based on sklearn, xgboost, and lightgbm.

1. Framework Connectors: Optimized, secure and performant data provisioning for Pytorch and Tensorflow frameworks in
their native data loader formats.

1. FileSet API: FileSet provides a Python fsspec-compliant API for materializing data into a Snowflake internal stage
from a query or Snowpark Dataframe along with a number of convenience APIs.

### Snowpark Model Management [Public Preview]

[Snowpark Model Management](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#snowpark-ml-ops) complements
the Snowpark ML Development API, and provides model management capabilities along with integrated deployment into Snowflake.
Currently, the API consists of:

1. Registry: A python API for managing models within Snowflake which also supports deployment of ML models into Snowflake
as native MODEL object running with Snowflake Warehouse.

## Getting started

### Have your Snowflake account ready

If you don't have a Snowflake account yet, you can [sign up for a 30-day free trial account](https://signup.snowflake.com/).

### Installation

Follow the [installation instructions](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index#installing-snowpark-ml)
in the Snowflake documentation.

Python versions 3.8 to 3.11 are supported. You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) or
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
