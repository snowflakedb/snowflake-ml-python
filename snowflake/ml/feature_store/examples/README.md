# Feature Store examples

## Example notebooks

You can find end-to-end demo notebooks from [Snowflake-Labs](https://github.com/Snowflake-Labs/snowflake-demo-notebooks).
Specifically these 4 notebooks use Feature Store:

- Feature Store Quickstart
- Feature Store API Overview
- End-to-end ML with Feature Store and Model Registry
- Manage features in DBT with Feature Store

## Example features

We have prepared some example features with publicly available datasets. The feature views, entities and datasets can
be easily loade by `ExampleHelper` (see below section). And some of notebooks, like *End-to-end ML with Feature Store
and Model Registry*, use these features as well.

<!-- markdownlint-disable -->
| Name                      | Data sources                                                              | Feature views                  |
| ------------------------  | ------------------------------------------------------------              | ----------------------------   |
| citibike_trip_features    | <https://www.kaggle.com/datasets/sujan97/citibike-system-data>              | trip_features               |
|                           |                                                                           | location_features                |
| new_york_taxi_features    | <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>              | station_feature    |
|                           |                                                                           | trip_feature     |
| wine_quality_features     | <https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>  | managed_wine_features          |
|                           |                                                                           | static_wine_features           |
| airline_features          | <s3://sfquickstarts/misc/demos/airline/>                                  | plane_features          |
|                           |                                                                           | weather_features           |
<!-- markdownlint-enable -->

## ExampleHelper

`ExampleHelper` is a helper class to load public datasets into Snowflake, load pre-defined feature views and entities.
It is available in `snowflake-ml-python` since 1.5.5.

```python
from snowflake.ml.feature_store.examples.example_helper import ExampleHelper

example_helper = ExampleHelper(session, <database_name>, <schema_name>)

# list all available examples in this directory.
example_helper.list_examples()

# load source data into Snowflake for a specific example.
source_tables = example_helper.load_example('citibike_trip_features')

# load draft entities for the selected example.
example_helper.load_entities()

# load draft feature views for the selected example.
example_helper.load_draft_feature_views()
```
