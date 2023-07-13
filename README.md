# Snowpark ML

Snowpark ML is a set of tools including SDKs and underlying infrastructure to build and deploy machine learning models. With Snowpark ML, you can pre-process data, train, manage and deploy ML models all within Snowflake, using a single SDK, and benefit from Snowflake’s proven performance, scalability, stability and governance at every stage of the Machine Learning workflow.

## Key Components of Snowpark ML
The Snowpark ML Python SDK provides a number of APIs to support each stage of an end-to-end Machine Learning development and deployment process, and includes two key components.

### Snowpark ML Development [Public Preview]

A collection of python APIs to enable efficient model development directly in Snowflake:

1. Modeling API (snowflake.ml.modeling) for data preprocessing, feature engineering and model training in Snowflake. This includes snowflake.ml.modeling.preprocessing for scalable data transformations on large data sets utilizing the compute resources of underlying Snowpark Optimized High Memory Warehouses, and a large collection of ML model development classes based on sklearn, xgboost, and lightgbm. See the private preview limited access docs (Preprocessing, Modeling for more details on these.

1. Framework Connectors: Optimized, secure and performant data provisioning for Pytorch and Tensorflow frameworks in their native data loader formats.

### Snowpark ML Ops [Private Preview]

Snowpark MLOps complements the Snowpark ML Development API, and provides model management capabilities along with integrated deployment into Snowflake. Currently, the API consists of
1. FileSet API: FileSet provides a Python fsspec-compliant API for materializing data into a Snowflake internal stage from a query or Snowpark Dataframe along with a number of convenience APIs.

1. Model Registry: A python API for managing models within Snowflake which also supports deployment of ML models into Snowflake Warehouses as vectorized UDFs.

During PrPr, we are iterating on API without backward compatibility guarantees. It is better to recreate your registry everytime you update the package. This means, at this time, you cannot use the registry for production use.

- [Documentation](https://docs.snowflake.com/developer-guide/snowpark-ml)

## Getting started
### Have your Snowflake account ready
If you don't have a Snowflake account yet, you can [sign up for a 30-day free trial account](https://signup.snowflake.com/).

### Create a Python virtual environment
Python 3.8 is required. You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html), [anaconda](https://www.anaconda.com/), or [virtualenv](https://docs.python.org/3/tutorial/venv.html) to create a Python 3.8 virtual environment.

To have the best experience when using this library, [creating a local conda environment with the Snowflake channel](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-packages.html#local-development-and-testing) is recommended.

### Install the library to the Python virtual environment
```
pip install snowflake-ml-python
```
