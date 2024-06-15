# Feature Store demo notebooks

Here we have 3 example notebooks that demonstrate Feature Store use cases in different scenarios.

## Basic Feature Demo

This example demonstrates creating and managing **non time-series features** in Feature Store and using the features in
training and inference. It also demonstrates interoperation with Model Registry. You can find the example in
[Basic_Feature_Demo.ipynb](https://github.com/snowflakedb/snowflake-ml-python/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/Basic_Feature_Demo.ipynb)
and [Basic_Feature_Demo.pdf](https://github.com/snowflakedb/snowflake-ml-python/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/Basic_Feature_Demo.pdf).

## Time Series Feature Demo

This example demonstrates creating and managing **time-series features** in Feature Store and using the features in
training and inference. It also demonstrates interoperation with Model Registry. You can find the example in
[Time_Series_Feature_Demo.ipynb](https://github.com/snowflakedb/snowflake-ml-python/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/DBT_External_Feature_Pipeline_Demo.ipynb)
and [Time_Series_Feature_Demo.pdf](https://github.com/snowflakedb/snowflake-ml-python/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/Time_Series_Feature_Demo.pdf).

## DBT External Feature Pipeline Demo

This example demonstrates how to register [DBT models](https://docs.getdbt.com/docs/build/models) as external Feature
Views in Feature Store. You need to have a DBT account to run this demo. You can find the example in
[DBT_External_Feature_Pipeline_Demo.ipynb](https://github.com/snowflakedb/snowflake-ml-python/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/DBT_External_Feature_Pipeline_Demo.ipynb)
and [DBT_External_Feature_Pipeline_Demo.pdf](https://github.com/snowflakedb/snowflake-ml-python/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/DBT_External_Feature_Pipeline_Demo.pdf).
