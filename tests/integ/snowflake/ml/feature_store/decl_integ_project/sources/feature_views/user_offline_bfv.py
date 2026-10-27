"""USER_OFFLINE_BFV offline tiled BatchFeatureView (Python authoring form).

Reads the same ``EVENTS_BATCH`` source as the online BFV but materialises
offline-only (``online=False``). It references the source by name string
(``SourceRef``) rather than a Python object because the ``BatchSource`` is
authored in YAML. The ``{{ suffix }}`` / ``{{ warehouse }}`` tokens are
substituted at plan time.
"""

from snowflake.ml.feature_store.decl import (
    BatchFeatureView,
    Feature,
    FSColumn,
    SourceRef,
)

user_offline_bfv = BatchFeatureView(
    name="USER_OFFLINE_BFV{{ suffix }}",
    version="V1",
    online=False,
    entities=["USER_ID"],
    sources=[SourceRef(name="EVENTS_BATCH{{ suffix }}", source_type="Batch")],
    timestamp_col="EVENT_TS",
    feature_granularity="1h",
    feature_aggregation_method="tiles",
    refresh_freq="5 minutes",
    warehouse="{{ warehouse }}",
    features=[
        Feature(
            source_column=FSColumn(name="CLICK_COUNT", type="IntegerType"),
            output_column=FSColumn(name="CLICK_SUM_1H", type="IntegerType"),
            function="sum",
            window="1h",
        ),
    ],
)
