# Feature Store examples

This folder contains some feature examples and the required source data.
All the source data are publicly available. You can easily get everything with `ExampleHelper`.

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

## Examples

Below table briefly describes all available examples. You can find all examples in this directory.

<!-- markdownlint-disable -->
| Name                      | Data sources                                                              | Feature views                  |
| ------------------------  | ------------------------------------------------------------              | ----------------------------   |
| citibike_trip_features    | <https://www.kaggle.com/datasets/sujan97/citibike-system-data>              | dropoff_features               |
|                           |                                                                           | pickup_features                |
| new_york_taxi_features    | <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>              | station_feature    |
|                           |                                                                           | trip_feature     |
| wine_quality_features     | <https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>  | managed_wine_features          |
|                           |                                                                           | static_wine_features           |
<!-- markdownlint-enable -->
