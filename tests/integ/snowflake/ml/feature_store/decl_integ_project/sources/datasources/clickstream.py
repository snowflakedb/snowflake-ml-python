"""CLICKSTREAM streaming source (Python authoring form).

The ``{{ suffix }}`` token is substituted at plan time by the integ test's
run-unique suffix so repeated runs against a reused schema never collide.
"""

from snowflake.ml.feature_store.decl import FSColumn, StreamingSource

clickstream = StreamingSource(
    name="CLICKSTREAM{{ suffix }}",
    type="REST",
    columns=[
        FSColumn(name="USER_ID", type="StringType"),
        FSColumn(name="SESSION_ID", type="StringType"),
        FSColumn(name="PAGE_URL", type="StringType"),
        FSColumn(name="EVENT_TYPE", type="StringType"),
        FSColumn(name="TIMESTAMP", type="TimestampType"),
        FSColumn(name="TIME_ON_PAGE_SECONDS", type="DoubleType"),
    ],
)
