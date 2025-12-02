from __future__ import annotations

import common_utils

from snowflake.snowpark import Session


def ensure_canonical_datasets(session: Session, db_name: str, schema_name: str) -> dict[str, object]:
    """Ensure wine and yellow taxi datasets exist in the given db.schema.

    Tries CTAS from a central canonical layout, else loads example datasets and
    creates compatibility views with canonical names/columns.

    Args:
        session: Active Snowpark session.
        db_name: Target database name where objects should be provisioned.
        schema_name: Target schema name under db_name.

    Returns:
        A dict with keys:
            - canonical (bool): True if CTAS from central succeeded.
            - wine_quality (str): FQN of the wine dataset table/view.
            - yellow_trip (str): FQN of the yellow trips table/view.

    Raises:
        AssertionError: If both CTAS and ExampleHelper provisioning fail.
    """

    yellow_name = common_utils.FS_INTEG_TEST_YELLOW_TRIP_DATA
    wine_name = common_utils.FS_INTEG_TEST_WINE_QUALITY_DATA

    target_yellow = f"{db_name}.{schema_name}.{yellow_name}"
    target_wine = f"{db_name}.{schema_name}.{wine_name}"

    result: dict[str, object] = {
        "canonical": False,
        "yellow_trip": target_yellow,
        "wine_quality": target_wine,
    }

    # Ensure target schema exists
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {db_name}.{schema_name}").collect()
    try:
        from snowflake.ml.feature_store.examples.example_helper import ExampleHelper

        helper = ExampleHelper(session, db_name, schema_name)
        try:
            helper.load_source_data("nyc_yellow_trips")
            helper.load_source_data("winequality_red")
        except Exception as e:
            raise AssertionError(
                "Failed to load example datasets via ExampleHelper. "
                "Ensure network access to public S3 and proper permissions. "
                f"Original error: {e}"
            )

        # Create compatibility views with canonical names and projected columns used by tests
        session.sql(
            f"CREATE OR REPLACE VIEW {target_yellow} AS "
            f"SELECT PULOCATIONID, TIP_AMOUNT, TOTAL_AMOUNT, TPEP_DROPOFF_DATETIME "
            f"FROM {db_name}.{schema_name}.nyc_yellow_trips"
        ).collect()
        session.sql(
            f"CREATE OR REPLACE VIEW {target_wine} AS "
            f"SELECT fixed_acidity, volatile_acidity, citric_acid, residual_sugar, "
            f"       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, "
            f"       sulphates, alcohol, quality "
            f"FROM {db_name}.{schema_name}.winedata"
        ).collect()
        return result
    except Exception as e:
        raise AssertionError(
            "Unable to provision datasets via CTAS or ExampleHelper. "
            "Tests expect either canonical clones or example-backed compatibility views. "
            f"Original error: {e}"
        )
