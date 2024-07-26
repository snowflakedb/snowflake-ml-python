import uuid

from absl.testing import absltest
from sklearn import datasets

from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView
from snowflake.ml.lineage.lineage_node import LineageNode
from snowflake.ml.model import ModelVersion
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.registry import Registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


class TestSnowflakeLineage(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        self._db_manager = db_manager.DBManager(self._session)
        self._current_db = self._session.get_current_database().replace('"', "")
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            uuid.uuid4().hex.upper()[:5], "schema"
        ).upper()
        self._db_manager.create_schema(self._test_schema)

    def tearDown(self) -> None:
        self._db_manager.drop_schema(self._test_schema)
        super().tearDown()

    def _create_iris_table(self, session: Session, iris_table: str) -> str:
        iris_df = datasets.load_iris(as_frame=True).frame
        iris_df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_df.columns.str.upper()]
        iris_df_snowflake = session.create_dataframe(iris_df)
        iris_df_snowflake.write.mode("overwrite").save_as_table(iris_table)

    def _check_lineage(self, res, expected_name, expected_version, expected_domain):
        assert len(res) == 1
        assert res[0]._lineage_node_name == expected_name
        assert res[0]._lineage_node_domain == expected_domain
        if expected_version:
            assert res[0]._lineage_node_version == expected_version

    def test_lineage(self):
        self._db_manager.use_schema(self._test_schema)
        table_name = f"IRIS_TABLE_{uuid.uuid4().hex.upper()[:5]}"
        feature_view_name = "IRIS_FEARURE_VIEW"
        feature_view_version = uuid.uuid4().hex.upper()[:5]
        dataset_name = '"iris_dataset"'
        dataset_version = "V" + uuid.uuid4().hex.upper()[:5]
        model_name = "IRIS_MODEL"
        model_version = "V" + uuid.uuid4().hex.upper()[:5]

        self._create_iris_table(self.session, f"{self._test_schema}.{table_name}")
        df = self._session.table(table_name)

        fs = FeatureStore(
            self._session,
            self._current_db,
            self._test_schema,
            default_warehouse=self._session.get_current_warehouse(),
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )

        e = Entity("iris_ds", ["TARGET"])
        fs.register_entity(e)

        fv = FeatureView(
            name=feature_view_name,
            entities=[e],
            feature_df=df,
            timestamp_col=None,
            desc="Iris dataset feature view",
        )
        fv = fs.register_feature_view(feature_view=fv, version=feature_view_version)

        spine_df = df.select("TARGET")

        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[fv],
            spine_timestamp_col=None,
            include_feature_view_timestamp_col=False,
            name=f"{self._test_schema}.{dataset_name}",
            version=dataset_version,
            output_type="dataset",
        )

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)

        regr.fit(ds.read.to_snowpark_dataframe())

        registry = Registry(session=self.session)
        model_ref = registry.log_model(
            model_name=f"{self._test_schema}.{model_name}", version_name=model_version, model=regr
        )

        # Case 1: Query lineage from feature view object
        fv_upstream = fv.lineage(direction="upstream")
        self._check_lineage(fv_upstream, f"{self._current_db}.{self._test_schema}.{table_name}", None, "table")
        assert isinstance(fv_upstream[0], LineageNode)

        fv_downstream = fv.lineage(domain_filter=["dataset"])
        self._check_lineage(
            fv_downstream, f"{self._current_db}.{self._test_schema}.{dataset_name}", dataset_version, "dataset"
        )
        assert isinstance(fv_downstream[0], LineageNode)

        from snowflake.ml import dataset

        fv_downstream = fv.lineage(domain_filter=["dataset"])
        self._check_lineage(
            fv_downstream, f"{self._current_db}.{self._test_schema}.{dataset_name}", dataset_version, "dataset"
        )
        assert isinstance(fv_downstream[0], dataset.Dataset)

        # Case 2 : Query lineage from dataset object
        ds_upstream = ds.lineage(direction="upstream", domain_filter=["feature_view"])
        self._check_lineage(
            ds_upstream,
            f"{self._current_db}.{self._test_schema}.{feature_view_name}",
            feature_view_version,
            "feature_view",
        )
        assert isinstance(ds_upstream[0], FeatureView)

        ds_downstream = ds.lineage()
        self._check_lineage(
            ds_downstream, f"{self._current_db}.{self._test_schema}.{model_name}", model_version, "model"
        )
        assert isinstance(ds_downstream[0], ModelVersion)

        # Case 3 : Query lineage from model object
        model_upstream = model_ref.lineage(direction="upstream")
        self._check_lineage(
            model_upstream, f"{self._current_db}.{self._test_schema}.{dataset_name}", dataset_version, "dataset"
        )
        assert isinstance(model_upstream[0], dataset.Dataset)

        # Case 4: lineage from Lineage nodes
        ds_upstream = model_upstream[0].lineage(direction="upstream", domain_filter=["feature_view"])
        self._check_lineage(
            ds_upstream,
            f"{self._current_db}.{self._test_schema}.{feature_view_name}",
            feature_view_version,
            "feature_view",
        )
        assert isinstance(ds_upstream[0], FeatureView)


if __name__ == "__main__":
    absltest.main()
