"""USER_FRAUD_FG FeatureGroup (Python authoring form).

Aggregates the online Postgres BatchFeatureView (``USER_AMOUNT_BFV``) into a
single FeatureGroup. The member is referenced by ``FeatureViewRef`` (name +
version strings) rather than a Python object because the member FV is authored
in YAML. ``{{ suffix }}`` is substituted at plan time and must match the
member FV's suffixed name.
"""

from snowflake.ml.feature_store.decl import FeatureGroup, FeatureViewRef

user_fraud_fg = FeatureGroup(
    name="USER_FRAUD_FG{{ suffix }}",
    version="V1",
    desc="User-level fraud-signal feature group (declarative integ).",
    auto_prefix=True,
    feature_views=[
        FeatureViewRef(name="USER_AMOUNT_BFV{{ suffix }}", version="V1"),
    ],
)
