import time

from snowflake.ml.feature_store._internal.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

if __name__ == "__main__":
    session = Session.builder.configs(SnowflakeLoginOptions()).create()
    db = "SNOWML_FEATURE_STORE_TEST_DB"
    schema = "TEST_DATASET"
    table = "tbao_test_data"

    df = session.table(table)
    print(df.to_pandas().describe())
    for s in df.schema:
        print(s)

    generator = SyntheticDataGenerator(session, db, schema, table)
    generator.trigger(3, 5)

    i = 0
    while True:
        if i > 60:
            break
        print(f"i: {i}")
        time.sleep(1)
        i = i + 1
